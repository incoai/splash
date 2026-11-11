#include "engine/Bootstrap.hpp"
#include "engine/Engine.hpp"
#include "engine/Json.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "Supervision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef SPLASH_BUILD_ID
#error "width probe requires the production build identity"
#endif

using namespace splash;
using namespace splash::engine;

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point value) {
  return std::chrono::duration<double, std::milli>(value.time_since_epoch()).count();
}

struct Case {
  std::string name;
  uint32_t cap = 0;
  std::vector<uint32_t> tokens;
};

Case readCase(const std::filesystem::path &path, const std::string &name) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open the original cohort");
  std::optional<Case> selected;
  for (std::string line; std::getline(input, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    Case value;
    uint64_t cap = 0, token = 0;
    if (!(row >> value.name >> cap) || !cap || cap > 8192)
      throw std::runtime_error("invalid cohort case or output cap");
    value.cap = static_cast<uint32_t>(cap);
    while (row >> token) {
      if (token > UINT32_MAX) throw std::runtime_error("cohort token overflow");
      value.tokens.push_back(static_cast<uint32_t>(token));
    }
    if (!row.eof() || value.tokens.empty())
      throw std::runtime_error("invalid cohort token list");
    if (value.name == name) {
      if (selected) throw std::runtime_error("cohort case is duplicated");
      selected = std::move(value);
    }
  }
  if (!selected) throw std::runtime_error("the requested case is absent");
  return std::move(*selected);
}

class Capture {
public:
  Capture(model::Runtime &runtime, const std::filesystem::path &directory,
          uint32_t vocabulary, uint32_t cap)
      : runtime_(runtime), vocabulary_(vocabulary), cap_(cap),
        metadata_(directory / "rows.jsonl"), binary_(directory / "logits.f32", std::ios::binary) {
    if (!metadata_ || !binary_) throw std::runtime_error("cannot open diagnostic output");
    model::RuntimeInstrumentation::captureTargetLogits(runtime_, true);
  }

  ~Capture() { model::RuntimeInstrumentation::captureTargetLogits(runtime_, false); }

  void batch(WorkKind kind, uint32_t width, uint32_t outputs, uint32_t accepted) {
    auto lanes = model::RuntimeInstrumentation::takeTargetLogits(runtime_);
    if (!lanes.empty()) {
      if (width != 1 || lanes.size() != 1 || lanes.front().size() % vocabulary_)
        throw std::runtime_error("capture requires one complete serial lane");
      const bool prefill = kind == WorkKind::Prefill;
      const uint32_t rows = std::min<uint32_t>(
          static_cast<uint32_t>(lanes.front().size() / vocabulary_), prefill ? 1 : accepted + 1);
      for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t index = prefill ? 0 : produced_ + 1 + row;
        if (index >= cap_) continue;
        const float *values = lanes.front().data() + uint64_t{row} * vocabulary_;
        if (!std::all_of(values, values + vocabulary_, [](float v) { return std::isfinite(v); }))
          throw std::runtime_error("captured target logits contain a nonfinite value");
        std::vector<uint32_t> ids(vocabulary_);
        std::iota(ids.begin(), ids.end(), 0u);
        const uint32_t top = std::min(vocabulary_, 8u);
        std::partial_sort(ids.begin(), ids.begin() + top, ids.end(), [&](uint32_t a, uint32_t b) {
          return values[a] != values[b] ? values[a] > values[b] : a < b;
        });
        predictions_.emplace_back(index, ids.front());
        const bool keep = index == 0 || (index >= 32 && index <= 48);
        json::Writer writer;
        writer.object([&] {
          writer.field("prediction_index", index).field("prefill", prefill)
              .field("output_count_before", produced_).field("batch_outputs", outputs)
              .field("accepted", accepted).field("batch_index", batches_).field("row", row)
              .field("vocabulary", vocabulary_).key("binary_offset");
          if (keep) writer.value(offset_); else writer.null();
          writer.array("top8", [&] {
            for (uint32_t i = 0; i < top; ++i)
              writer.array([&] { writer.value(ids[i]).value(values[ids[i]]); });
          });
        });
        metadata_ << writer.str() << '\n';
        if (keep) {
          binary_.write(reinterpret_cast<const char *>(values), uint64_t{vocabulary_} * sizeof(float));
          offset_ += uint64_t{vocabulary_} * sizeof(float);
        }
      }
    }
    produced_ += outputs;
    ++batches_;
    metadata_.flush();
    binary_.flush();
    if (!metadata_ || !binary_) throw std::runtime_error("diagnostic write failed");
  }

  void validate(const std::vector<uint32_t> &returned) const {
    bool covered42 = returned.size() <= 42;
    for (const auto &[index, argmax] : predictions_) {
      if (index < returned.size() && returned[index] != argmax)
        throw std::runtime_error("captured argmax differs from returned token at " + std::to_string(index));
      if (index == 42) covered42 = true;
    }
    if (!covered42) throw std::runtime_error("the capture missed output index 42");
  }

private:
  model::Runtime &runtime_;
  uint32_t vocabulary_, cap_, produced_ = 0;
  uint64_t offset_ = 0, batches_ = 0;
  std::ofstream metadata_, binary_;
  std::vector<std::pair<uint32_t, uint32_t>> predictions_;
};

class Events final : public EngineEventSink {
public:
  Capture *capture = nullptr;
  std::string cacheStatus, failure;
  uint32_t matchedTokens = 0;
  bool done = false;
  std::vector<uint32_t> returned;

  void batchCompleted(WorkKind kind, uint32_t width, uint32_t, uint32_t outputs,
                      uint32_t, uint32_t accepted, double, double) override {
    if (capture) capture->batch(kind, width, outputs, accepted);
  }
  void started(uint64_t, uint32_t matched, uint32_t) override {
    matchedTokens = matched;
    cacheStatus = matched ? "prefix_hit" : "miss";
  }
  void tokens(uint64_t, std::span<const uint32_t> tokens) override {
    returned.insert(returned.end(), tokens.begin(), tokens.end());
  }
  void maskRequested(uint64_t, std::span<const uint32_t>) override {
    throw std::runtime_error("a greedy unconstrained probe requested a mask");
  }
  void completed(uint64_t, EngineFinishReason, bool, uint32_t, uint32_t, uint32_t,
                 std::span<const float>, const SpeculativeCounts &) override { done = true; }
  void failed(uint64_t, LaneOutcome, std::string message, uint32_t) override { failure = std::move(message); }
};

class Driver {
public:
  explicit Driver(Engine &engine) : engine_(engine) {
    engine_.setCompletionNotifier([wake = wake_] {
      {
        std::lock_guard lock(wake->mutex);
        wake->notified = true;
      }
      wake->condition.notify_one();
    });
  }
  void untilIdle() {
    const auto deadline = Clock::now() + std::chrono::hours(2);
    while (true) {
      const auto snapshot = engine_.snapshot();
      if (snapshot.submitted == snapshot.completed + snapshot.cancelled + snapshot.failed &&
          !engine_.commandInFlight()) return;
      if (engine_.tick(milliseconds(Clock::now()))) continue;
      if (Clock::now() >= deadline) throw std::runtime_error("probe request timed out");
      auto wakeup = deadline;
      if (auto next = engine_.nextWakeupMilliseconds())
        wakeup = std::min(wakeup, Clock::time_point{
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(*next))});
      std::unique_lock lock(wake_->mutex);
      wake_->condition.wait_until(lock, wakeup, [this] {
        if (!wake_->notified) return false;
        wake_->notified = false;
        return true;
      });
    }
  }
private:
  struct Wake { std::mutex mutex; std::condition_variable condition; bool notified = false; };
  Engine &engine_;
  std::shared_ptr<Wake> wake_ = std::make_shared<Wake>();
};

void evictCache(Cache &cache) {
  bool gave;
  do {
    MemoryLedger ledger;
    ledger.addSource(cache);
    const ClaimId claim = ledger.open({.cls = ClaimClass::InService, .reach = ReclaimClass::InUse});
    gave = ledger.free(claim, {.bytes = 1}, 0, false).gave();
    ledger.close(claim);
  } while (gave);
  static_cast<void>(cache.releaseEmptyExtents(false));
  const CacheSnapshot snapshot = cache.snapshot();
  if (snapshot.stateCache.entries || snapshot.pool.pagesPrefix)
    throw std::runtime_error("probe cache did not drain");
}

void warm(RuntimeResources &resources, model::RuntimeModel &runtime) {
  MemoryGovernor &governor = resources.memoryGovernor();
  const ClaimId claim = governor.ledger().open({.cls = ClaimClass::Floor});
  {
    MemoryGovernor::Scope scope(governor, claim);
    KvPool::RunwayHold runway(resources.kvPool());
    static_cast<void>(runtime.warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
    for (uint32_t lanes = 1; lanes <= model::ExecutionLimits::maximumBatchWidth; ++lanes)
      static_cast<void>(runtime.warmupDecodeBatch(lanes));
  }
  governor.ledger().close(claim);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 7) {
      std::cerr << "usage: width-probe METALLIB MODEL_ROOT COHORT CASE ROWS OUTPUT_DIR\n";
      return 2;
    }
    const std::string rowsText = argv[5];
    if (rowsText != "8" && rowsText != "16") throw std::invalid_argument("ROWS must be 8 or 16");
    const uint32_t rows = rowsText == "16" ? 16 : 8;
    const Case selected = readCase(argv[3], argv[4]);
    const std::filesystem::path output = argv[6];
    if (!std::filesystem::create_directory(output))
      throw std::runtime_error("OUTPUT_DIR must be a new directory with an existing parent");
    RuntimeBootstrapConfig config;
    config.resources.metallibPath = argv[1];
    config.resources.modelRoot = argv[2];
    config.resources.model = model::servedFor(model::inspectModelRoot(argv[2]), 32768);
    config.resources.buildId = SPLASH_BUILD_ID;
    config.resources.aneFfn = AneFfnSetting::fromGiven(0.0, {});
    config.resources.verifyRows = model::VerifyRows::All;
    config.resources.draftBlockSize = rows;
    config.nativeLoop.engine.maxContext = 32768;
    auto bootstrap = ServingRuntime::start(config, [](std::span<const uint8_t>) {}, [] { return 0.0; });
    bootstrap->announceReady();
    if (!bootstrap->nativeLoop().engineHealthy() || bootstrap->nativeLoop().commandInFlight())
      throw std::runtime_error("production bootstrap is not idle and healthy");
    auto &resources = bootstrap->resources();
    auto &runtime = bootstrap->modelRuntime();
    auto *concrete = dynamic_cast<model::Runtime *>(&runtime);
    if (!concrete || runtime.decodeRows() != rows) throw std::runtime_error("unexpected runtime geometry");
    warm(resources, runtime);
    evictCache(resources.cache());
    Events events;
    EngineConfig engineConfig;
    engineConfig.maxContext = bootstrap->nativeLoop().statusSnapshot().maximumContextTokens;
    engineConfig.vocabularySize = config.resources.model.capabilities.vocabularySize;
    engineConfig.draftContextTokens = config.resources.model.draft.contextTokens;
    connectToGovernor(engineConfig, resources.memoryGovernor());
    Engine engine(engineConfig, resources.cache(), runtime, events);
    Driver driver(engine);
    Capture capture(*concrete, output, engineConfig.vocabularySize, selected.cap);
    events.capture = &capture;
    EngineRequest request;
    request.id = 1;
    request.prompt = selected.tokens;
    request.maxNewTokens = selected.cap;
    request.sampling.temperature = 0.0F;
    request.sampling.topP = 1.0F;
    request.deadlineMilliseconds = milliseconds(Clock::now() + std::chrono::hours(2));
    try {
      engine.submit(std::move(request));
      driver.untilIdle();
    } catch (...) {
      events.capture = nullptr;
      throw;
    }
    events.capture = nullptr;
    if (!events.done || !events.failure.empty() || events.cacheStatus != "miss" || events.matchedTokens)
      throw std::runtime_error("cold probe failed or reused a prefix: " + events.failure);
    capture.validate(events.returned);
    json::Writer writer;
    writer.object([&] {
      writer.field("case", selected.name).field("build_id", SPLASH_BUILD_ID)
          .field("verify_rows", rows).field("model_root", std::string(argv[2]))
          .field("loaded_model_layout_sha256", resources.cacheIdentity().modelLayoutSha256)
          .field("device", resources.backend().capabilities().deviceName)
          .field("cohort", std::string(argv[3])).field("output_cap", selected.cap)
          .field("context_limit", engineConfig.maxContext).field("temperature", 0)
          .field("ane_ffn_share", resources.aneFfnShare())
          .field("cache_status", events.cacheStatus).field("matched_tokens", events.matchedTokens)
          .field("all_captured_argmax_mappings_match", true);
      writer.array("prompt_tokens", [&] { for (uint32_t token : selected.tokens) writer.value(token); });
      writer.array("output_tokens", [&] { for (uint32_t token : events.returned) writer.value(token); });
      writer.field("scope", "One cold greedy serial request with host copies of pre-policy target logits. "
          "Full-vocabulary rows are retained at output index 0 and indices 32 through 48. "
          "The original cohort cap and runtime decode path are preserved. Diagnostic timings are unused.");
    });
    std::ofstream result(output / "result.json");
    result << writer.str() << '\n';
    if (!result) throw std::runtime_error("cannot write the probe result");
    std::cout << writer.str() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "width-probe: " << error.what() << '\n';
    return 1;
  }
}
