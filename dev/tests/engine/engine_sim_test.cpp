// Integration scenarios: the native loop and its engine on the fakes, at a
// Mac's fitted timing profile and memory. CPU only; every input is fixed, so
// a run repeats itself exactly.

#include "EngineHarness.hpp"
#include "FakeElasticPool.hpp"
#include "FakeMemory.hpp"
#include "LandingUnit.hpp"
#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"
#include "TimingProfile.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash;

namespace {

using test::require;

constexpr uint64_t kSlotBytes = 1'382'400;  // a Flash-Next expert as the store streams it
constexpr uint32_t kTokensPerCommand = 8;   // DFlash2 commits 7.93-7.98 tokens a lane command (TimingProfile.cpp)

// --- The machine ----------------------------------------------------------------------------------------------

// The expert store as a pool of whole chunks (FakeElasticPool).
struct Store final {
  uint64_t unitBytes = 0;
  uint32_t floor = 0, cap = 0, whole = 0;
};

struct Machine final {
  const char *name = "";
  const test::TimingProfile *profile = nullptr;
  // The governor's limit, the reserves it leaves untracked, and what the
  // runtime holds beside the KV, the states and the store: fixed_runtime_bytes
  // less those reserves and the store at its cap.
  uint64_t limitBytes = 0, untrackedBytes = 0, fixedBytes = 0;
  // The Mac's memory, what other applications held at the run's first sample
  // (physical less allocated less spare), and the host reserves.
  uint64_t hostBytes = 0, otherBytes = 0, hostReserveBytes = 0, hostSpareReserveBytes = 0;
  uint64_t pageBytes = 0, indexPageBytes = 0;
  uint32_t extentPages = 256;
  // A lane's state buffers: lane_state_bytes over the four lanes it plans.
  uint64_t laneStateBytes = 0;
  // A decoding Spill lane's floor of the hot block cache.
  uint64_t laneHotFloorBytes = 0;
  Store store;
  engine::SpillGeometry spill;
  // A prefill command's rows, in up to a pass a layer.
  uint32_t commandRows = model::ExecutionLimits::prefillTokenBudget;
};

// The Studio (36 GB) at --max-memory 18G serving Flash-Next Q2_0, its numbers
// as /status reports them. Its store chunk of 954 slots puts the floor and cap
// at 5724 and 6678 slots, the whole chunks nearest the recorded 5672 and 6679.
const Machine kStudio18G{
    .name = "studio-18g",
    .profile = &test::kStudio18GFlashNextQ2,
    .limitBytes = 19'327'352'832,
    .untrackedBytes = 805'306'368,
    .fixedBytes = 17'048'316'698 - 805'306'368 - 7ULL * 954 * kSlotBytes,
    .hostBytes = 38'654'705'664,
    .otherBytes = 38'654'705'664 - 16'091'482'912 - 12'678'086'656,
    .hostReserveBytes = 2'147'483'648,
    .hostSpareReserveBytes = 2'705'829'392,
    .pageBytes = 423'936,
    .indexPageBytes = 24'576,
    .laneStateBytes = 276'496'384 / 4,
    .laneHotFloorBytes = 408'944'640,
    .store = {.unitBytes = 954 * kSlotBytes, .floor = 6, .cap = 7, .whole = 26},
    .spill = {.fullPageBytes = 423'936,
              .indexPageBytes = 24'576,
              .slotBytes = 100,  // the test tier's slots (TestKvTier::slotBytes), as its quota counts them
              .residentTokens = 147'449,
              .densePages = 64,
              .windowPages = 64,
              .groupPages = 256},
    .commandRows = 8192,
};

// Qwen3.6-35B-A3B at --max-memory 18G on the Studio, streaming its experts
// without HiSparse: no Index pages, no Spill, the store's floor 10 of its 12
// chunks (0.85 of the cap, whole chunks).
const Machine k35BStudio18G{
    .name = "studio-18g-35b",
    .profile = &test::k35BStudio18G,
    .limitBytes = 19'327'352'832,
    .untrackedBytes = 805'306'368,
    .fixedBytes = 18'378'746'256 - 805'306'368 - 12ULL * 560 * 1'911'603,
    .hostBytes = 38'654'705'664,
    .otherBytes = 38'654'705'664 - 16'091'482'912 - 12'678'086'656,
    .hostReserveBytes = 2'147'483'648,
    .hostSpareReserveBytes = 2'705'829'392,
    .pageBytes = 332'800,
    .laneStateBytes = 181'469'184 / 4,
    .store = {.unitBytes = 560 * 1'911'603, .floor = 10, .cap = 12, .whole = 18},
    .commandRows = 8192,
};

std::string at(const Machine &machine) { return std::string(machine.name) + ": "; }

// A scenario's numbers, beside its verdict.
void note(const std::string &line) { std::printf("engine-sim: %s\n", line.c_str()); }

// --- Requests -----------------------------------------------------------------------------------------------

// A prompt of `tokens` tokens no other prompt shares a block of.
std::vector<uint32_t> prompt(uint64_t seed, uint32_t tokens) {
  std::vector<uint32_t> result(tokens);
  for (uint32_t index = 0; index < tokens; ++index)
    result[index] = static_cast<uint32_t>(seed << 20 | index);
  return result;
}

protocol::RequestFrame frame(uint64_t id, std::vector<uint32_t> tokens, uint32_t output,
                             engine::RequestPriority priority = engine::RequestPriority::Normal) {
  protocol::RequestFrame result;
  result.requestId = id;
  result.priority = priority;
  result.constraint = ConstraintMode::None;
  // An hour, on the fixture's unix clock (1 s).
  result.absoluteDeadlineUnixMicros = 1'000'000 + 3'600'000'000ULL;
  result.remainingDeadlineMicros = 3'600'000'000ULL;
  result.logicalMaxOutputTokens = output;
  result.promptTokens = std::move(tokens);
  return result;
}

// Three workers' prompts: 59,043 tokens first and 12,200 more a round, the
// workers a few tokens apart; every fourth round resends the previous
// conversation and 8 tokens more. Output: 1504 tokens, whole commands.
struct ChurnPrompts final {
  uint32_t first = 59'043, step = 12'200;
  std::array<int32_t, 3> offsets{-23, 11, 0};
  uint32_t output = 1504;
  [[nodiscard]] uint32_t tokens(uint32_t worker, uint32_t round) const {
    return static_cast<uint32_t>(int64_t{first} + int64_t{step} * (round - 1) + offsets[worker]);
  }
};

// --- A loop on the fakes at a machine's memory ----------------------------------------------------------------

// The native loop (LoopFixture) at `machine`'s memory and profile: FakeMemory
// holding the fixed runtime, the governor at the machine's limit and host,
// the store as a FakeElasticPool, the KV tier and spill plan, and the
// accounts and pools a start books.
struct Serving final {
  explicit Serving(const Machine &machine, std::vector<test::HostEvent> script = {})
      : machine(machine),
        fixture(config(machine), shape(machine), {}, metal::kResidencyKeepAliveSeconds, options(machine, script)) {
    // The fixture's weights stand for part of the fixed runtime.
    const uint64_t weights = fixture.memory->held("weights");
    require(machine.fixedBytes >= weights, at(machine) + "the fixed runtime is less than the fixture's weights");
    fixture.memory->hold("runtime", machine.fixedBytes - weights);
    experts.emplace("experts", *fixture.memory, machine.store.unitBytes, machine.store.floor, machine.store.cap,
                    machine.store.whole, machine.store.cap);
    test::Executor &model = fixture.executor;
    model.experts = &*experts;
    model.decodeTokens = kTokensPerCommand;
    model.decodeFinishes = false;
    model.firstTokens = true;
    model.laneFootprintBytes = machine.laneStateBytes;
    model.spilledReadFloorBytes = machine.laneHotFloorBytes;
    // The history staging of the floor's 8192 pages, 281,149,440 B.
    model.stagingPageBytes = 281'149'440 / 8192;
    engine::MemoryLedger &ledger = fixture.governor->ledger();
    static_cast<void>(ledger.addAccount("kv", [this] { return fixture.memory->held("kv"); }));
    static_cast<void>(ledger.addAccount("states", [this] { return fixture.memory->held("states"); }));
    static_cast<void>(ledger.addPool(*experts));
    model.commandTimed = [this](const BatchPlan &, const test::CommandTimes &times) {
      lastCommandEnd = fixture.monotonic + times.wallMilliseconds;
    };
  }

  static engine::NativeLoopConfig config(const Machine &machine) {
    engine::NativeLoopConfig result;
    result.engine.maxContext = 262'144;
    result.engine.spill = {.capable = machine.spill.fullPageBytes != 0,
                           .denseTokens = machine.spill.densePages * engine::KvCache::pageTokens,
                           .geometry = machine.spill};
    return result;
  }
  // Pages for the whole limit, in the machine's extents.
  static test::PoolShape shape(const Machine &machine) {
    const auto extents = static_cast<uint32_t>(machine.limitBytes / machine.pageBytes / machine.extentPages + 1);
    return {.pages = extents * machine.extentPages,
            .extentPages = machine.extentPages,
            .runwayPages = 0,
            .pageBytes = machine.pageBytes,
            .indexPageBytes = machine.indexPageBytes};
  }
  static test::LoopOptions options(const Machine &machine, std::vector<test::HostEvent> script) {
    test::LoopOptions result{
        .memory = test::FakeMemory::Config{.hostBytes = machine.hostBytes,
                                           .otherBytes = machine.otherBytes,
                                           .script = std::move(script)},
        .limits = {.limitBytes = machine.limitBytes,
                   .hostReserveBytes = machine.hostReserveBytes,
                   .hostSpareReserveBytes = machine.hostSpareReserveBytes,
                   .untrackedReserveBytes = machine.untrackedBytes},
        .timing = machine.profile,
        .shape = [rows = machine.commandRows, profile = machine.profile](test::Executor &model) {
          model.groupRows = rows;
          model.passes = profile->layers;
          model.streamedExperts = test::streamsExperts(*profile);
        }};
    // A quota past any churn's KV; 64 transfers of a page each in flight land
    // at the profile's SSD rate together (TestKvTier's slots are 100 bytes).
    result.tier = test::TierOptions{.capacity = 4'000'000,
                                    .transferLimit = 64,
                                    .bytesPerMillisecond = 100.0 * machine.profile->weightBytesPerMillisecond /
                                                           (64.0 * double(machine.pageBytes))};
    if (machine.spill.fullPageBytes)
      result.cacheSpill = engine::CacheSpillPolicy{machine.spill.densePages, machine.spill.windowPages};
    return result;
  }

  // Sends a request now.
  void submit(uint64_t id, std::vector<uint32_t> tokens, uint32_t output,
              engine::RequestPriority priority = engine::RequestPriority::Normal) {
    submittedAt[id] = fixture.monotonic;
    require(fixture.loop.receive(protocol::peer::serialize(frame(id, std::move(tokens), output, priority))),
            "a request frame was refused");
  }
  [[nodiscard]] bool active(uint64_t id) const { return submittedAt.contains(id) && !ended.contains(id); }
  // Reads what the loop wrote since the last step: each request's end.
  void drain() {
    if (fixture.output.empty())
      return;
    for (const protocol::EngineEvent &event : reader.feed(fixture.output)) {
      if (const auto *done = std::get_if<protocol::DoneEvent>(&event)) {
        ended.insert(done->requestId);
        endings[done->requestId] = {done->completionTokens, done->reason};
      } else if (const auto *error = std::get_if<protocol::ErrorEvent>(&event)) {
        ended.insert(error->requestId);
        errors.push_back(std::to_string(error->requestId) + " " + error->code + ": " + error->message);
      }
    }
    fixture.output.clear();
  }
  // Requests are in flight, yet no command has run for a resource wait since
  // `since` or the last command's end, and no claim waits that the engine
  // would still fail in time.
  [[nodiscard]] bool stalled(double since) const {
    since = std::max(since, lastCommandEnd);
    if (fixture.loop.commandInFlight() || fixture.monotonic - since <= engine::kResourceWaitTimeoutMilliseconds ||
        std::ranges::none_of(submittedAt, [&](const auto &request) { return active(request.first); }))
      return false;
    const engine::MemoryClaimsSnapshot claims = memory().claims;
    return !claims.waiting || claims.oldestMilliseconds > engine::kResourceWaitTimeoutMilliseconds;
  }
  // Ticks until done(), observing each step, failing with `what` once the
  // clock passes `until` or the loop stalls.
  void run(const std::function<bool()> &done, const std::function<void()> &observe, std::string_view what,
           std::optional<double> until = std::nullopt) {
    const double since = fixture.monotonic;
    fixture.run(
        [&] {
          drain();
          if (observe)
            observe();
          return done() || (until && fixture.monotonic >= *until) || stalled(since);
        },
        "the loop did not move", std::numeric_limits<uint32_t>::max());
    require(done(), at(machine) + std::string(what) + " at " + std::to_string(fixture.monotonic / 1000.0) + " s" +
                        holders());
  }
  // What FakeMemory holds by holder, and the requests' phases, for a
  // failure's message.
  [[nodiscard]] std::string holders() const {
    std::string result = " (MiB held:";
    for (const char *holder : {"runtime", "experts", "kv", "states", "spilled_reads", "weights"})
      result += std::string(" ") + holder + " " + std::to_string(fixture.memory->held(holder) >> 20);
    const engine::MemoryGovernorSnapshot governor = memory();
    result += "; charged " + std::to_string(governor.chargedBytes >> 20) + " of " +
              std::to_string(governor.limitBytes >> 20) + "; requests:";
    for (const engine::ActiveRequestSnapshot &request : fixture.loop.statusSnapshot().activeRequests)
      result += " " + std::to_string(request.id) + ":" + std::to_string(int(request.phase)) + "/" +
                std::to_string(int(request.kvMode)) + "/" + std::to_string(request.promptProcessed) + "of" +
                std::to_string(request.promptTokens);
    return result + ")";
  }
  [[nodiscard]] engine::MemoryGovernorSnapshot memory() const { return fixture.governor->snapshot(); }
  [[nodiscard]] double now() const noexcept { return fixture.monotonic; }

  const Machine &machine;
  test::LoopFixture fixture;
  std::optional<test::FakeElasticPool> experts;
  double lastCommandEnd = 0.0;
  std::map<uint64_t, double> submittedAt;
  std::set<uint64_t> ended;
  // How each request that completed ended: its output and finish reason.
  std::map<uint64_t, std::pair<uint32_t, engine::EngineFinishReason>> endings;
  std::vector<std::string> errors;
  protocol::peer::EventReader reader;
};

// Three closed-loop workers arriving in `order` 0.6 s apart, until
// `endMilliseconds`; each step observed.
struct Churn final {
  Churn(Serving &serving, std::array<uint32_t, 3> order) : serving(serving) {
    for (uint32_t rank = 0; rank < 3; ++rank)
      next[order[rank]] = serving.now() + 600.0 * rank;
  }
  void run(double endMilliseconds, const std::function<void()> &observe) {
    while (serving.now() < endMilliseconds) {
      for (uint32_t worker = 0; worker < 3; ++worker) {
        if (inFlight[worker] && !serving.active(*inFlight[worker])) {
          finished.push_back(*inFlight[worker]);
          inFlight[worker].reset();
          next[worker] = serving.now();
        }
        if (!inFlight[worker] && next[worker] <= serving.now()) {
          const uint32_t round = ++rounds[worker];
          const uint64_t id = 100 * (worker + 1) + round;
          std::vector<uint32_t> tokens;
          if (round % 4 == 0 && !last[worker].empty()) {
            tokens = last[worker];
            const std::vector<uint32_t> again = prompt(9000 + id, 8);
            tokens.insert(tokens.end(), again.begin(), again.end());
          } else {
            tokens = prompt(id, prompts.tokens(worker, round));
            last[worker] = tokens;
          }
          serving.submit(id, std::move(tokens), prompts.output);
          inFlight[worker] = id;
        }
      }
      const double nextArrival = std::min({next[0], next[1], next[2]});
      serving.run(
          [&] {
            if (serving.now() >= endMilliseconds)
              return true;
            for (uint32_t worker = 0; worker < 3; ++worker)
              if ((inFlight[worker] && !serving.active(*inFlight[worker])) ||
                  (!inFlight[worker] && next[worker] <= serving.now()))
                return true;
            return !inFlight[0] && !inFlight[1] && !inFlight[2] && serving.now() >= nextArrival;
          },
          observe, "the churn did not move", endMilliseconds);
      require(serving.errors.empty(), at(serving.machine) + "a churn request failed: " +
                                          (serving.errors.empty() ? std::string() : serving.errors.front()) +
                                          serving.holders());
    }
  }

  Serving &serving;
  ChurnPrompts prompts;
  std::array<double, 3> next{};
  std::array<uint32_t, 3> rounds{};
  std::array<std::optional<uint64_t>, 3> inFlight{};
  std::array<std::vector<uint32_t>, 3> last{};
  std::vector<uint64_t> finished;
};

// One decode command of `decoders` lanes at the experts' `share`,
// `spillLanes` of them reading spilled K/V.
double decodeCommand(const test::TimingProfile &profile, uint32_t decoders, double share, uint32_t spillLanes) {
  return test::decodeTimes(profile, decoders, decoders * model::ExecutionLimits::targetVerifyRows, share, spillLanes)
      .wallMilliseconds;
}

// --- Scenarios -----------------------------------------------------------------------------------------------

// A 15-minute churn through a host episode short of critical pressure keeps
// the store, turning it at most once a host level, with bounded gaps and waits.
void testChurnKeepsTheStoreThroughAHostEpisode() {
  const Machine &machine = kStudio18G;
  constexpr double episode = 300'000.0, warningEnd = 420'000.0, release = 600'000.0, end = 900'000.0;
  constexpr double episodeEnd = release + 60'000.0;
  // Other applications leave the host a store chunk short of its spare
  // reserve: warned from 300 to 420 s, given back at 600 s.
  const uint64_t started = machine.fixedBytes + uint64_t{machine.store.cap} * machine.store.unitBytes;
  const uint64_t episodeOther = machine.hostBytes - started - machine.hostSpareReserveBytes + machine.store.unitBytes;
  Serving serving(machine, {{episode, episodeOther, engine::MemoryPressure::Warning},
                            {warningEnd, episodeOther, engine::MemoryPressure::Normal},
                            {release, machine.otherBytes, engine::MemoryPressure::Normal}});
  Churn churn(serving, {2, 0, 1});
  uint64_t last = serving.experts->state().currentBytes;
  int direction = 0;
  std::map<int, uint32_t> turnsByLevel;  // per level of the host: 1 warning, 2 short, 3 the minute after
  bool episodeShrank = false;
  double oldestClaim = 0.0;
  churn.run(end, [&] {
    const double now = serving.now();
    const uint64_t current = serving.experts->state().currentBytes;
    if (current != last) {
      const int moved = current > last ? 1 : -1;
      episodeShrank |= moved < 0 && now >= episode && now < release;
      if (now >= episode && now < episodeEnd && direction && moved != direction)
        ++turnsByLevel[now < warningEnd ? 1 : now < release ? 2 : 3];
      direction = moved;
      last = current;
    }
    oldestClaim = std::max(oldestClaim, serving.memory().claims.oldestMilliseconds);
  });
  const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
  const double gapLimit = counts.scheduler.decodeGapTargetMilliseconds +
                          decodeCommand(*machine.profile, 3, double(machine.store.floor) / machine.store.whole, 3);
  const double longestGap = serving.fixture.loop.metricsSnapshot().decodeGap.maximumMilliseconds;
  require(churn.finished.size() >= 6, at(machine) + "the churn finished " + std::to_string(churn.finished.size()) +
                                          " requests in 15 minutes");
  require(longestGap <= gapLimit, at(machine) + "a decode gap of " + std::to_string(longestGap) +
                                      " ms passed the run's bound and a decode command, " + std::to_string(gapLimit) +
                                      " ms");
  require(oldestClaim < engine::kResourceWaitTimeoutMilliseconds && counts.failed == 0,
          at(machine) + "a request waited " + std::to_string(oldestClaim) + " ms for memory, or failed");
  for (const auto &[level, turns] : turnsByLevel)
    require(turns <= 1, at(machine) + "the store changed direction " + std::to_string(turns) +
                            " times at one host level (" + std::to_string(level) + ")");
  require(!episodeShrank, at(machine) + "the host's episode took a store unit short of critical pressure");
  note(at(machine) + std::to_string(churn.finished.size()) + " requests, longest decode gap " +
       std::to_string(longestGap) + " ms of " + std::to_string(gapLimit));
}

// The engine harness running `profile`'s commands on its clock, a pass a
// layer, the first token at the prompt's end; with `memory`, under a
// governor over FakeMemory.
struct TimedHarness final {
  explicit TimedHarness(const test::TimingProfile &profile, std::optional<test::FakeMemory::Config> memory = {})
      : harness({.pages = 8192,
                 .engine = {.maxContext = 262'144},
                 .memory = std::move(memory),
                 .timing = &profile,
                 .shape = [&profile](test::Executor &model) {
                   model.groupRows = model::ExecutionLimits::prefillTokenBudget;
                   model.passes = profile.layers;
                   model.streamedExperts = test::streamsExperts(profile);
                 }}) {
    harness.model.decodeTokens = kTokensPerCommand;
    harness.model.decodeFinishes = false;
    harness.model.firstTokens = true;
    harness.model.time(profile);
  }
  void submit(uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    engine::EngineRequest request = harness.request(id, std::move(tokens));
    request.maxNewTokens = output;
    request.deadlineMilliseconds = harness.now() + 3'600'000.0;
    submitted[id] = harness.now();
    harness.engine.submit(std::move(request));
  }
  [[nodiscard]] std::optional<double> ttft(uint64_t id) const {
    const auto found = harness.events.tokenTimes.find(id);
    if (found == harness.events.tokenTimes.end() || found->second.empty())
      return std::nullopt;
    return found->second.front() - submitted.at(id);
  }

  test::EngineHarness harness;
  std::map<uint64_t, double> submitted;
};

// The 27B's 64K prompt beside two decoders: their gaps stay within the unit
// bound and a decode command; the prompt slows at most kLongPromptSlowdown.
constexpr double kLongPromptSlowdown = 1.08;
void testLongPromptBesideTwoDecoders() {
  const test::TimingProfile &profile = test::kDense27B;
  // The prompt's prefill time, its decode share taken out.
  const auto prefillOf = [&](bool decoders) {
    TimedHarness served(profile);
    if (decoders) {
      served.submit(1, prompt(1, 1024), 60'000);
      served.submit(2, prompt(2, 1024), 60'000);
      served.harness.runUntil([&] { return served.ttft(1) && served.ttft(2); }, "the decoders did not start");
    }
    const size_t first = served.harness.commands().size();
    served.submit(3, prompt(3, 65'536), kTokensPerCommand);
    served.harness.runUntil([&] { return served.ttft(3).has_value(); }, "the long prompt did not prefill");
    double decode = 0.0, floorUnit = 0.0;
    for (size_t index = first; index < served.harness.commands().size(); ++index) {
      const test::EngineHarness::Command &command = served.harness.commands()[index];
      uint32_t rows = 0;
      for (const uint32_t lane : command.rows)
        rows += lane;
      if (command.kind == WorkKind::Decode)
        decode += command.wallMilliseconds;
      else if (command.kind == WorkKind::Prefill && rows <= 512)
        floorUnit = std::max(floorUnit, command.wallMilliseconds);
    }
    if (decoders) {
      const double bound = std::max(served.harness.engine.snapshot().scheduler.decodeGapTargetMilliseconds, floorUnit) +
                           decodeCommand(profile, 2, 1.0, 0);
      for (const uint64_t decoder : {1u, 2u})
        require(served.harness.longestDecodeGap(decoder) <= bound,
                "a decoder's gap beside the long prompt, " + std::to_string(served.harness.longestDecodeGap(decoder)) +
                    " ms, passed " + std::to_string(bound) + " ms");
    }
    return *served.ttft(3) - decode;
  };
  const double alone = prefillOf(false), opposed = prefillOf(true);
  require(opposed <= kLongPromptSlowdown * alone, "the long prompt's prefill beside two decoders took " +
                                                       std::to_string(opposed) + " ms of its own against " +
                                                       std::to_string(alone) + " ms alone");
  note("the 27B's 64K prompt prefilled in " + std::to_string(opposed) + " ms of its own beside two decoders, " +
       std::to_string(alone) + " ms alone");
}

// Three 60K prompts at once, 2,000 tokens of output each, with the model's states written behind to the
// SSD cache as the 35B's are; runs until they end, observing each step, and gives their prompts.
std::vector<std::vector<uint32_t>> burst(Serving &serving, const std::function<void()> &observe) {
  serving.fixture.executor.keptTier = std::make_shared<test::OffloadControl>();
  serving.fixture.executor.keptTier->ready = true;
  std::vector<std::vector<uint32_t>> prompts;
  for (uint64_t id = 1; id <= 3; ++id) {
    prompts.push_back(prompt(id, 60'000));
    serving.submit(id, prompts.back(), 2'000);
  }
  serving.run([&] { return serving.ended.size() == 3; }, observe, "the burst did not end");
  require(serving.errors.empty(), at(serving.machine) + "a request of the burst failed");
  return prompts;
}

// The block each prompt's resume point sits on, none without one. A probe leaves the LRU as it is.
std::vector<std::optional<uint64_t>> resumeBlocks(Serving &serving,
                                                  const std::vector<std::vector<uint32_t>> &prompts) {
  std::vector<std::optional<uint64_t>> result;
  for (const std::vector<uint32_t> &tokens : prompts) {
    const engine::CacheProbe probe = serving.fixture.cache.probe(tokens, {});
    const std::span<const uint64_t> chain = probe.restoredChain();
    result.push_back(chain.empty() ? std::nullopt : std::optional<uint64_t>(chain.back()));
  }
  return result;
}

// The 35B (store floor 0.85 of its cap) after a burst of three 60K prompts that took a store chunk: the
// chunk comes back no sooner than a quiet period after the burst took it and within a minute of the
// last request's end, from cache that frees now (copied or dead cache, and the one state the idle
// staging buffer writes): no demotion starts for it, the newest conversation's resume point stays in
// RAM, and the copies written ahead while idle stay within the chunk.
void testRegrowthAfterABurstTakesOnlyCacheThatFreesNow() {
  const Machine &machine = k35BStudio18G;
  Serving serving(machine);
  std::vector<std::pair<double, uint32_t>> moves;
  const auto observe = [&] {
    if (const uint32_t units = serving.experts->units(); moves.empty() ? units != machine.store.cap
                                                                        : units != moves.back().second)
      moves.emplace_back(serving.now(), units);
  };
  const std::vector<std::vector<uint32_t>> prompts = burst(serving, observe);
  require(moves.size() == 1 && moves.front().second == machine.store.cap - 1,
          at(machine) + "the burst did not take exactly one store chunk");
  const double taken = moves.front().first, ended = serving.now();
  const std::optional<uint64_t> newest = resumeBlocks(serving, prompts).back();
  require(newest && serving.fixture.cache.stateResident(*newest),
          at(machine) + "the burst did not leave its newest conversation a resume point in RAM");
  const engine::CacheSnapshot idle = serving.fixture.cache.snapshot();
  serving.run([&] { return serving.now() >= ended + 90'000.0; }, observe, "the idle time did not pass");
  const engine::CacheSnapshot regrown = serving.fixture.cache.snapshot();
  const double copiedBytes = double(regrown.kvTier.copies - idle.kvTier.copies) * double(machine.pageBytes);
  require(moves.size() == 2 && moves.back().second == machine.store.cap &&
              moves.back().first >= taken + 1000.0 * engine::kRegrowQuietSeconds &&
              moves.back().first <= ended + 60'000.0,
          at(machine) + "the store did not regrow its chunk between a quiet period after the burst took it and " +
              "a minute after the last request");
  require(serving.fixture.cache.stateResident(*newest) && regrown.kvTier.demotions == idle.kvTier.demotions &&
              copiedBytes <= double(machine.store.unitBytes),
          at(machine) + "regrowth took the newest resume point, demoted cache to take it, or copied more than " +
              "a chunk ahead");
  note(at(machine) + "the burst took a chunk at " + std::to_string(taken / 1000.0) + " s, the last request ended at " +
       std::to_string(ended / 1000.0) + " s, the chunk came back at " + std::to_string(moves.back().first / 1000.0) +
       " s after " + std::to_string(copiedBytes / double(1 << 20)) + " MiB of copies while idle");
}

// Critical pressure after the same burst, macOS's level sampled every half second for 6 s: each settled
// second takes about the margin, cache that frees now before the store's chunk above its floor, never
// the store below its floor, and the newest resume point stays in RAM while older cache covers the
// margin.
void testCriticalTakesTheMarginASecondAndKeepsTheResumePoint() {
  const Machine &machine = k35BStudio18G;
  constexpr double critical = 66'000.0, normal = 72'000.0;
  std::vector<test::HostEvent> script;
  for (double at = critical; at < normal; at += 500.0)
    script.push_back({at, machine.otherBytes, engine::MemoryPressure::Critical});
  script.push_back({normal, machine.otherBytes, engine::MemoryPressure::Normal});
  Serving serving(machine, std::move(script));
  const std::vector<std::vector<uint32_t>> prompts = burst(serving, {});
  const auto held = [&] {
    return serving.fixture.memory->held("kv") + serving.fixture.memory->held("states") +
           uint64_t{serving.experts->units()} * machine.store.unitBytes;
  };
  const std::optional<uint64_t> newest = resumeBlocks(serving, prompts).back();
  require(serving.now() < critical && serving.experts->units() == machine.store.floor + 1 && newest &&
              serving.fixture.cache.stateResident(*newest),
          at(machine) + "the burst did not leave a store chunk above the floor and a resume point in RAM");
  // What each critical pass took, of the cache, the states and the store,
  // and whether the newest resume point stayed in RAM.
  struct Pass final {
    double at = 0.0;
    uint64_t bytes = 0;
    uint32_t units = 0;
    bool resumePoint = false;
  };
  std::vector<Pass> passes;
  uint64_t last = held();
  uint32_t lowest = serving.experts->units();
  serving.run([&] { return serving.now() >= normal + 5'000.0; }, [&] {
    lowest = std::min(lowest, serving.experts->units());
    if (const uint64_t now = held(); now < last) {
      passes.push_back(
          {serving.now(), last - now, serving.experts->units(), serving.fixture.cache.stateResident(*newest)});
      last = now;
    }
  }, "the critical episode did not pass");
  require(passes.size() >= 2 && std::ranges::all_of(passes, [&](const Pass &pass) {
            return pass.at >= critical && pass.at < normal && pass.bytes < 2 * engine::kHostMarginBytes;
          }) && passes[0].bytes >= engine::kHostMarginBytes &&
              passes[1].at - passes[0].at >= engine::kHostSettleMilliseconds,
          at(machine) + "a critical pass took more or less than about the margin, or two ran within a settled second");
  require(passes[0].units == machine.store.floor + 1 && lowest == machine.store.floor,
          at(machine) + "critical pressure took the store before the cache that frees now, or below its floor");
  require(passes[0].resumePoint,
          at(machine) + "critical pressure took the newest resume point while older cache covered the margin");
  std::string taken;
  for (const Pass &pass : passes)
    taken += " " + std::to_string(pass.bytes >> 20) + " MiB at " + std::to_string(pass.at / 1000.0) + " s" +
             (pass.resumePoint ? "" : ", the resume point with it") + ";";
  note(at(machine) + "critical passes took" + taken);
}

// A younger short prompt, as a resend's uncached rows, starts beside a
// running prefill that leads an older, longer prompt refused memory: the
// refusal holds back only the starts the scheduler wants after it, and the
// refused prompt's claim holds no room while the scheduler does not want it.
void testAResendStartsBesideARefusedLongerPrompt() {
  test::LandingUnit landing{1ULL << 30};
  TimedHarness served(test::kDense27B, test::FakeMemory::Config{});
  test::EngineHarness &harness = served.harness;
  engine::MemoryLedger &ledger = harness.governor->ledger();
  harness.model.laneFootprintBytes = 64 << 10;
  const auto waiting = [&](uint64_t id) {
    return std::ranges::any_of(ledger.claims(), [&](const auto &entry) {
      return entry.second.spec.requestId == id && entry.second.waiting;
    });
  };
  const auto phase = [&](uint64_t id) {
    for (const engine::ActiveRequestSnapshot &active : harness.engine.snapshot().activeRequests)
      if (active.id == id)
        return active.phase;
    return engine::Phase::Completed;
  };
  // P (30K) finds no free lane at its first attempt, so R (45K), which
  // arrived after it, is tried alone and refused memory that lands later.
  harness.model.maximumLanes = 0;
  served.submit(1, prompt(1, 30'000), kTokensPerCommand);
  static_cast<void>(harness.tick());
  harness.model.maximumLanes = model::ExecutionLimits::maximumBatchWidth;
  harness.memory->hold("limit", harness.governor->snapshot().headroomBytes - 32);
  ledger.addSource(landing);
  served.submit(2, prompt(2, 45'000), kTokensPerCommand);
  static_cast<void>(harness.tick());
  require(waiting(2) && !harness.model.requests.contains(1) && !harness.model.requests.contains(2),
          "the longer prompt was not refused memory that lands later");
  harness.memory->release("limit", harness.memory->held("limit"));
  landing.landed = true;
  harness.runUntil([&] { return harness.model.requests.contains(1); }, "P did not start at its retry");
  // S (1.5K) arrives while P prefills.
  served.submit(3, prompt(3, 1536), kTokensPerCommand);
  bool heldRoom = false;
  harness.runUntil(
      [&] {
        heldRoom = heldRoom || (waiting(2) && phase(2) != engine::Phase::WaitingResources);
        return test::idle(harness.engine);
      },
      "the prompts did not finish");
  const auto first = [&](uint64_t id) { return harness.events.tokenTimes.at(id).front(); };
  require(first(3) < first(1),
          "the younger short prompt waited for P's prefill behind the refused one: its first token at " +
              std::to_string(first(3) / 1000.0) + " s, P's at " + std::to_string(first(1) / 1000.0) + " s");
  require(!heldRoom, "the refused prompt's claim held room while the scheduler did not want it");
  require(first(2) > first(1) && harness.events.completedCount == 3 && harness.events.failedCount == 0,
          "the refused prompt did not start after P, or a prompt failed");
  note("the 27B's 1.5K prompt answered " + std::to_string(*served.ttft(3) / 1000.0) +
       " s after it arrived beside P's 30K prefill and a refused 45K prompt; P answered at " +
       std::to_string(*served.ttft(1) / 1000.0) + " s");
}

// The 35B on the Studio: eight closed-loop workers resend their ~8.6K-token
// conversations, cached but for their last rows, while a 38,133-token prompt
// nothing caches waits. Each command that serves the resends runs a few
// hundred rows, yet the cold prompt waits behind at most
// K = maximumBatchWidth - 1 prefill commands that serve later arrivals, then
// takes the next lane that frees and leads.
void testAColdPromptStartsWithinTheOvertakesAmidCachedResends() {
  constexpr uint32_t kWorkers = 8, kOutput = 64, kOvertakes = model::ExecutionLimits::maximumBatchWidth - 1;
  TimedHarness served(test::k35BStudio18G);
  test::EngineHarness &harness = served.harness;
  uint64_t nextId = 1;
  std::array<std::vector<uint32_t>, kWorkers> conversation;
  std::array<uint64_t, kWorkers> inFlight{};
  std::array<uint32_t, kWorkers> rounds{};
  const auto send = [&](uint32_t worker) {
    const uint64_t id = nextId++;
    const std::vector<uint32_t> turn = prompt(id, conversation[worker].empty() ? 8600 + 60 * worker : 8);
    conversation[worker].insert(conversation[worker].end(), turn.begin(), turn.end());
    served.submit(id, conversation[worker], kOutput);
    inFlight[worker] = id;
    ++rounds[worker];
  };
  // Runs the workers, each sending its next turn as its last one ends, until
  // `until`.
  const auto work = [&](const std::function<bool()> &until, std::string_view what) {
    const auto ended = [&](uint32_t worker) { return harness.events.usage.contains(inFlight[worker]); };
    while (!until()) {
      harness.runUntil(
          [&] { return until() || std::ranges::any_of(std::views::iota(0u, kWorkers), ended); }, what);
      for (uint32_t worker = 0; worker < kWorkers; ++worker)
        if (ended(worker))
          send(worker);
    }
  };
  for (uint32_t worker = 0; worker < kWorkers; ++worker)
    send(worker);
  work([&] { return std::ranges::all_of(rounds, [](uint32_t round) { return round >= 3; }); },
       "the workers did not finish their first turns");
  const uint64_t cold = nextId++;
  const size_t from = harness.commands().size();
  const double arrived = harness.now();
  served.submit(cold, prompt(cold, 38'133), kTokensPerCommand);
  // The prefill commands, first passes, from its arrival to its first one.
  uint32_t overtakes = 0, cachedResends = 0;
  size_t seen = from;
  bool led = false;
  const auto first = [&] {
    for (; !led && seen < harness.commands().size(); ++seen) {
      const test::EngineHarness::Command &command = harness.commands()[seen];
      if (command.kind != WorkKind::Prefill || command.pass)
        continue;
      led = std::ranges::count(command.lanes, cold) != 0;
      overtakes += !led && std::ranges::any_of(command.lanes, [&](uint64_t id) { return id > cold; });
    }
    return led || harness.now() >= arrived + 300'000.0;
  };
  work(first, "the workers and the cold prompt did not move");
  for (size_t index = 0; index < harness.events.startIds.size(); ++index)
    cachedResends += harness.events.startIds[index] > cold && harness.events.starts[index] >= 8000;
  require(cachedResends >= kOvertakes, "the workers' resends did not start from their cached conversations");
  require(led && overtakes <= kOvertakes,
          "the cold 38,133-token prompt " +
              (led ? "led after " + std::to_string((harness.now() - arrived) / 1000.0) + " s"
                   : std::string("had not led in 300 s")) +
              ", behind " + std::to_string(overtakes) + " prefill commands that served later arrivals, past K = " +
              std::to_string(kOvertakes));
  note("the 35B's cold 38,133-token prompt led " + std::to_string((harness.now() - arrived) / 1000.0) +
       " s after it arrived, behind " + std::to_string(overtakes) + " commands of " + std::to_string(cachedResends) +
       " cached resends");
}

// Guard: random schedules of four requests on the Studio at 1 GiB less than 18G: prompts of 80K to 110K tokens, which
// memory holds only in part, as a conversation tree (each of the later three, in half the cases, an earlier one's
// conversation up to a random page of it and 2,000 to 20,000 tokens of its own), outputs of 64 to 1,504 tokens, each
// priority, arrivals within 2 s, and three seconds of macOS's critical pressure between the first minute and the
// third; then a short Normal turn on the first conversation's opening pages, 10 to 60 s after it arrived, which
// placement may start Resident on a Spill lane's prefix. Spill lanes keep their pages Full, so the lanes' growth and
// reads, the starts and the host's critical passes release the pages whose copies landed, between commands. Nothing
// breaks an engine rule (std::logic_error), the cache's invariants hold after every step, no claim's count of what it
// takes covered its need where its takes then did not (MemoryLedger::CountChecks), no lane yields outside the
// critical pressure and the 5 s after it where none yields at twice the limit, and each request ends as it does where
// the limit and the host hold every request, at twice the limit: with the same output and finish reason (the fake
// model's tokens do not depend on the K/V it reads). SPLASH_SIM_RELEASE_SEEDS runs more seeds than the four.
void testRandomSchedulesWithReleasesEndAsWithoutThem() {
  Machine tight = kStudio18G;
  tight.name = "studio-17g";
  tight.limitBytes -= 1ULL << 30;
  Machine ample = kStudio18G;
  ample.name = "studio-ample";
  ample.limitBytes *= 2;
  ample.hostBytes *= 2;
  uint64_t releases = 0;
  const char *seeds = std::getenv("SPLASH_SIM_RELEASE_SEEDS");
  const uint32_t count = seeds ? static_cast<uint32_t>(std::stoul(seeds)) : 4;
  for (uint32_t seed = 1; seed <= count; ++seed) {
    std::mt19937 random(seed);
    const auto pick = [&](uint32_t first, uint32_t last) {
      return std::uniform_int_distribution<uint32_t>(first, last)(random);
    };
    struct Arrival final {
      double at = 0.0;
      uint64_t id = 0;
      std::vector<uint32_t> tokens;
      uint32_t output = 0;
      engine::RequestPriority priority = engine::RequestPriority::Normal;
    };
    std::vector<Arrival> schedule;
    for (uint64_t id = 1; id <= 4; ++id) {
      Arrival arrival{.at = double(pick(0, 20)) * 100.0,
                      .id = id,
                      .tokens = prompt(1'000 * seed + id, pick(80'000, 110'000)),
                      .output = 8 * pick(8, 188),
                      .priority = static_cast<engine::RequestPriority>(pick(0, 2))};
      if (id > 1 && pick(0, 1)) {
        const Arrival &parent = schedule[pick(0, static_cast<uint32_t>(id) - 2)];
        const uint32_t pages = pick(1, static_cast<uint32_t>(parent.tokens.size() / engine::KvCache::pageTokens));
        arrival.at = std::max(arrival.at, parent.at);
        std::vector<uint32_t> tokens(parent.tokens.begin(),
                                     parent.tokens.begin() + pages * engine::KvCache::pageTokens);
        const std::vector<uint32_t> more = prompt(9'000 + 10 * seed + id, pick(2'000, 20'000));
        tokens.insert(tokens.end(), more.begin(), more.end());
        arrival.tokens = std::move(tokens);
      }
      schedule.push_back(std::move(arrival));
    }
    Arrival turn{.at = schedule.front().at + double(pick(10, 60)) * 1'000.0, .id = 5, .output = 64};
    turn.tokens.assign(schedule.front().tokens.begin(),
                       schedule.front().tokens.begin() + pick(32, 256) * engine::KvCache::pageTokens);
    const std::vector<uint32_t> question = prompt(9'500 + seed, 1'000);
    turn.tokens.insert(turn.tokens.end(), question.begin(), question.end());
    schedule.push_back(std::move(turn));
    std::ranges::sort(schedule, {}, &Arrival::at);
    const double critical = double(pick(60, 180)) * 1'000.0;
    const std::vector<test::HostEvent> script{
        {critical, kStudio18G.otherBytes, engine::MemoryPressure::Critical},
        {critical + 3'000.0, kStudio18G.otherBytes, engine::MemoryPressure::Normal}};
    // The tight run's yields outside the critical pressure and the 5 s after it.
    size_t calmYields = 0;
    const auto run = [&](const Machine &machine) {
      Serving serving(machine, &machine == &tight ? script : std::vector<test::HostEvent>{});
      size_t next = 0, yields = 0;
      calmYields = 0;
      const auto observe = [&] {
        if (!serving.fixture.loop.commandInFlight())
          serving.fixture.cache.checkInvariants();
        const size_t now = serving.fixture.executor.suspendedIds.size();
        if (now != yields && (serving.now() < critical || serving.now() > critical + 8'000.0))
          calmYields += now - yields;
        yields = now;
      };
      while (serving.ended.size() < schedule.size()) {
        for (; next < schedule.size() && schedule[next].at <= serving.now(); ++next)
          serving.submit(schedule[next].id, schedule[next].tokens, schedule[next].output, schedule[next].priority);
        const double arrival = next < schedule.size() ? schedule[next].at : std::numeric_limits<double>::infinity();
        const size_t ended = serving.ended.size();
        serving.run([&] { return serving.ended.size() != ended || serving.now() >= arrival; }, observe,
                    "the schedule did not move", 1'800'000.0);
        require(serving.now() < 1'800'000.0, at(machine) + "seed " + std::to_string(seed) +
                                                  ": the requests did not end within 30 minutes");
      }
      require(serving.errors.empty(), at(machine) + "seed " + std::to_string(seed) + ": a request failed: " +
                                          (serving.errors.empty() ? std::string() : serving.errors.front()));
      const engine::MemoryLedger::CountChecks checks = serving.fixture.governor->snapshot().countChecks;
      require(checks.overcountedRuns == 0,
              at(machine) + "seed " + std::to_string(seed) + ": " + std::to_string(checks.overcountedRuns) + " of " +
                  std::to_string(checks.coveredRuns) + " claims whose count covered their need took less");
      releases += serving.fixture.cache.spillSnapshot().releases;
      return serving.endings;
    };
    const auto ends = run(tight);
    const size_t tightYields = calmYields;
    require(ends == run(ample), "seed " + std::to_string(seed) +
                                    ": a request ended otherwise than where the limit and the host hold them all");
    require(calmYields != 0 || tightYields == 0,
            "seed " + std::to_string(seed) + ": " + std::to_string(tightYields) +
                " lanes yielded at the tight limit outside the critical pressure, none at twice the limit");
  }
  require(releases != 0, "no schedule released a lane's pages");
  note("random schedules with " + std::to_string(releases) + " releases ended as where nothing was short");
}

} // namespace

std::span<const test::TestCase> simTests() {
  static constexpr test::TestCase kTests[] = {
      {"testLongPromptBesideTwoDecoders", testLongPromptBesideTwoDecoders},
      {"testChurnKeepsTheStoreThroughAHostEpisode", testChurnKeepsTheStoreThroughAHostEpisode},
      {"testRegrowthAfterABurstTakesOnlyCacheThatFreesNow", testRegrowthAfterABurstTakesOnlyCacheThatFreesNow},
      {"testCriticalTakesTheMarginASecondAndKeepsTheResumePoint",
       testCriticalTakesTheMarginASecondAndKeepsTheResumePoint},
      {"testAResendStartsBesideARefusedLongerPrompt", testAResendStartsBesideARefusedLongerPrompt},
      {"testAColdPromptStartsWithinTheOvertakesAmidCachedResends",
       testAColdPromptStartsWithinTheOvertakesAmidCachedResends},
      {"testRandomSchedulesWithReleasesEndAsWithoutThem", testRandomSchedulesWithReleasesEndAsWithoutThem},
  };
  return kTests;
}

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"sim", simTests()}};
  return test::runTests("engine-sim", groups, argc, argv, [] { test::metalStatistics() = {}; });
}
