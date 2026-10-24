#include "engine/Bootstrap.hpp"
#include "StderrLine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

namespace splash::engine {
namespace {

// macOS counts the memory Metal frees as available only a moment later, 0.5
// to 1 s after Metal's own count drops on a 24 GB Mac mini. A start's last
// check of the host's reserve waits up to this long, checking again at this
// interval, for what the start gave back: the lanes its warmups left idle
// and the state of a warmup the host refused.
constexpr auto kMemoryReleaseWait = std::chrono::seconds(2);
constexpr auto kMemoryReleasePoll = std::chrono::milliseconds(100);

RuntimeBootstrapReport reportForPlan(const EngineMemoryPlan &plan) {
  RuntimeBootstrapReport report;
  report.budgetDescription = plan.breakdown().describe();
  return report;
}

RuntimeBootstrapReport
reportForResourceFailure(const RuntimeResourcesError &error) {
  RuntimeBootstrapReport report;
  report.resourceStage = error.stage();
  report.resourceFailure = error.failure();
  report.message = error.what();
  report.budgetDescription = error.budgetDescription();
  return report;
}

[[noreturn]] void fail(RuntimeBootstrapReport report,
                       RuntimeBootstrapStage stage, std::string message) {
  report.stage = stage;
  report.message = std::move(message);
  throw RuntimeBootstrapError(std::move(report));
}

} // namespace

std::string_view runtimeBootstrapStageName(RuntimeBootstrapStage stage) {
  switch (stage) {
  case RuntimeBootstrapStage::ResourceAssembly:
    return "resource_assembly";
  case RuntimeBootstrapStage::ModelCreation:
    return "model_creation";
  case RuntimeBootstrapStage::MaximumPrefill:
    return "maximum_prefill";
  case RuntimeBootstrapStage::DecodeWarmup:
    return "decode_warmup";
  case RuntimeBootstrapStage::CompositeStateRestore:
    return "composite_state_restore";
  case RuntimeBootstrapStage::MemoryAudit:
    return "memory_audit";
  case RuntimeBootstrapStage::AnnounceReady:
    return "announce_ready";
  case RuntimeBootstrapStage::Ready:
    return "ready";
  }
  return "unknown";
}

std::string RuntimeBootstrapReport::describe() const {
  std::ostringstream out;
  out << (stage == RuntimeBootstrapStage::Ready ? "runtime bootstrap ready"
                                                : "runtime bootstrap failed")
      << " [" << runtimeBootstrapStageName(stage);
  if (resourceStage)
    out << '/' << runtimeResourceStageName(*resourceStage);
  out << "]: " << message;
  if (!memoryAudit.message.empty()) {
    out << '\n' << memoryAudit.describe();
  }
  if (!budgetDescription.empty())
    out << '\n' << budgetDescription;
  return out.str();
}

RuntimeBootstrapError::RuntimeBootstrapError(RuntimeBootstrapReport report)
    : std::runtime_error(report.describe()), report_(std::move(report)) {}

RuntimeBootstrapError::RuntimeBootstrapError(const RuntimeResourcesError &error)
    : RuntimeBootstrapError(reportForResourceFailure(error)) {}

std::optional<StartupRetryWindow::Clock::time_point>
StartupRetryWindow::retryUntil(const RuntimeBootstrapReport &failure,
                               Clock::time_point now) {
  if (failure.resourceFailure != RuntimeResourceFailure::HostCapacity &&
      failure.resourceFailure != RuntimeResourceFailure::DriverAllocation)
    return std::nullopt;
  const std::pair reached{failure.stage, failure.resourceStage};
  if (!deadline_ || reached > reached_) {
    deadline_ = now + length_;
    reached_ = reached;
  }
  if (now >= *deadline_)
    return std::nullopt;
  return deadline_;
}

protocol::ProtocolLimits protocolLimitsFor(
    const model::ModelCapabilities &capabilities, uint32_t maxContext) noexcept {
  protocol::ProtocolLimits limits;
  limits.maxPromptTokens = maxContext;
  limits.maxLogicalOutputTokens = maxContext;
  limits.maxTokenBatch = model::ExecutionLimits::maximumStepTokens;
  limits.maxSimulationTokens = model::ExecutionLimits::draftQueryRows;
  limits.maxMaskWords = model::maskWordsPerToken(capabilities.vocabularySize) *
                        (model::ExecutionLimits::draftQueryRows + 1);
  return limits;
}

RuntimeBootstrap::RuntimeBootstrap(std::unique_ptr<RuntimeResources> resources,
                                   std::unique_ptr<model::RuntimeModel> modelRuntime,
                                   std::unique_ptr<NativeRuntime> nativeLoop,
                                   RuntimeBootstrapReport report)
    : resources_(std::move(resources)), model_(std::move(modelRuntime)),
      nativeLoop_(std::move(nativeLoop)),
      memoryControl_(resources_->memoryGovernor(), resources_->backend(),
                     *nativeLoop_),
      report_(std::move(report)) {
  sampleThermalState();
}

RuntimeBootstrap::~RuntimeBootstrap() {
  // Refuse new commands before the loop, the model and the resources they
  // reach are destroyed.
  resources_->backend().stop();
}

std::string RuntimeBootstrap::statusJson(const RuntimeMetricsSnapshot &metrics,
                                         const NativeLoopTiming &loop) {
  // Status can arrive during GPU work; allocation/command boundaries and
  // the safe-point pressure monitor already refresh the cached sample.
  metal::MetalBackend &backend = resources_->backend();
  const bool healthy = backend.healthy();
  return runtimeStatusJson(
      resources_->memoryPlan(), nativeLoop_->statusSnapshot(), backend.memoryStats(),
      report_.warmup, report_.memoryAudit, metrics, model_->telemetry(),
      resources_->cacheIdentity(), resources_->memoryGovernor().snapshot(),
      healthy, healthy ? std::string{} : backend.unhealthyReason(),
      nativeLoop_->resourceWaitSnapshot(), loop, nativeLoop_->weightsSnapshot(),
      resources_->aneFfnSnapshot(), thermalState_.state(), resources_->streamedTableStats(),
      resources_->streamedExpertStats(), resources_->kvSpillStatus());
}

void RuntimeBootstrap::sampleThermalState() {
  const std::string transition = thermalState_.update(queryThermalState());
  if (!transition.empty())
    logLine(transition);
}

RuntimeBootstrapReport RuntimeBootstrap::requireWarmupAndAnnounce(
    const EngineMemoryPlan &memoryPlan, model::RuntimeModel &modelRuntime,
    ActualMemoryReporter memoryReporter, NativeRuntime &nativeLoop) {
  RuntimeBootstrapReport report = reportForPlan(memoryPlan);
  auto run = [&](RuntimeBootstrapStage stage, WarmupStepStatus &status,
                 auto &&operation, bool optional = false) {
    model::WarmupStepResult result;
    try {
      result = operation();
    } catch (const metal::MetalAllocationError &error) {
      if (optional) {
        status = WarmupStepStatus::MemoryLimited;
        return model::WarmupStepResult{};
      }
      report.resourceFailure = resourceAllocationFailure(error.failure());
      fail(report, stage,
           std::string(runtimeBootstrapStageName(stage)) +
               " threw: " + error.what());
    } catch (const std::exception &error) {
      fail(report, stage,
           std::string(runtimeBootstrapStageName(stage)) +
               " threw: " + error.what());
    } catch (...) {
      fail(report, stage,
           std::string(runtimeBootstrapStageName(stage)) +
               " threw an unknown exception");
    }
    if (!(result.wallSeconds > 0.0) || !std::isfinite(result.wallSeconds)) {
      std::string message = std::string(runtimeBootstrapStageName(stage)) +
                            " did not complete a real measured path";
      if (!result.detail.empty())
        message += ": " + result.detail;
      fail(report, stage, std::move(message));
    }
    status = WarmupStepStatus::Complete;
    return result;
  };

  model::WarmupStepResult maximumPrefill =
      run(RuntimeBootstrapStage::MaximumPrefill, report.warmup.maximumPrefill,
          [&] {
            return modelRuntime.warmupPrefill(model::ExecutionLimits::prefillTokenBudget);
          });
  nativeLoop.observePrefill(model::ExecutionLimits::prefillTokenBudget,
                           maximumPrefill.wallSeconds * 1000.0, maximumPrefill.hostSeconds * 1000.0);
  report.warmup.maximumPrefillDetail = maximumPrefill.detail;
  const auto &budget = memoryPlan.breakdown();
  // Startup exercises only widths that fit this budget. This is not a
  // serving concurrency limit: the engine still admits lanes dynamically.
  const uint32_t affordableWidth = static_cast<uint32_t>(std::min<uint64_t>(
      model::ExecutionLimits::maximumBatchWidth,
      (budget.dynamicBudgetBytes -
       kvRunwayPages(budget.kvExtentPages) * budget.kvPageBytes) /
          budget.laneStateBytes));
  for (uint32_t width = 1;
       width <= model::ExecutionLimits::maximumBatchWidth; ++width) {
    if (width > affordableWidth ||
        (width > 1 && report.warmup.decodeBatches[width - 2] ==
                          WarmupStepStatus::MemoryLimited)) {
      report.warmup.decodeBatches[width - 1] = WarmupStepStatus::MemoryLimited;
      continue;
    }
    run(RuntimeBootstrapStage::DecodeWarmup,
        report.warmup.decodeBatches[width - 1],
        [&] { return modelRuntime.warmupDecodeBatch(width); }, width > 1);
  }
  run(RuntimeBootstrapStage::CompositeStateRestore,
      report.warmup.compositeStateRestore,
      [&] { return modelRuntime.warmupCompositeStateRestore(); }, true);

  ActualMemoryReport actual;
  try {
    actual = memoryReporter();
  } catch (const metal::MetalAllocationError &error) {
    report.resourceFailure = resourceAllocationFailure(error.failure());
    fail(report, RuntimeBootstrapStage::MemoryAudit,
         std::string("actual memory reporting failed: ") + error.what());
  } catch (const std::exception &error) {
    fail(report, RuntimeBootstrapStage::MemoryAudit,
         std::string("actual memory reporting failed: ") + error.what());
  } catch (...) {
    fail(report, RuntimeBootstrapStage::MemoryAudit,
         "actual memory reporting failed with an unknown exception");
  }
  report.memoryAudit = auditActualMemory(memoryPlan, actual);
  if (!report.memoryAudit.valid) {
    fail(report, RuntimeBootstrapStage::MemoryAudit,
         report.memoryAudit.describe());
  }

  try {
    nativeLoop.announceReady();
  } catch (const std::exception &error) {
    fail(report, RuntimeBootstrapStage::AnnounceReady,
         std::string("ReadyEvent announcement failed: ") + error.what());
  } catch (...) {
    fail(report, RuntimeBootstrapStage::AnnounceReady,
         "ReadyEvent announcement failed with an unknown exception");
  }
  report.stage = RuntimeBootstrapStage::Ready;
  report.message = "required warmup paths and memory audit passed";
  return report;
}

std::unique_ptr<RuntimeBootstrap> RuntimeBootstrap::start(
    RuntimeBootstrapConfig config,
    NativeRuntime::ByteSink output,
    NativeRuntime::StatusProvider statusProvider) {
  std::unique_ptr<RuntimeResources> resources;
  try {
    resources = RuntimeResources::create(config.resources, config.nativeLoop.engine.maxContext);
  } catch (const RuntimeResourcesError &error) {
    throw RuntimeBootstrapError(error);
  }

  RuntimeBootstrapReport base = reportForPlan(resources->memoryPlan());
  const uint32_t planContext =
      resources->memoryPlan().maximumContextTokens();
  if (!planContext) {
    fail(std::move(base), RuntimeBootstrapStage::ModelCreation,
         "memory plan cannot hold one model token");
  }
  const uint32_t relaunchContext = config.resources.relaunchContext;
  if (!config.nativeLoop.engine.maxContext && relaunchContext) {
    // A relaunch serves the context of the server's first engine, which its
    // split left. A plan that holds less, as once this Mac's memory budget
    // shrank, has the relaunch serve that, and the server, which serves the
    // first engine's context, stops relaunching (server/runtime.py).
    if (relaunchContext > planContext)
      logLine("The memory plan holds ", planContext, " tokens of context, less than the ",
              relaunchContext, " the server's first engine served.");
    config.nativeLoop.engine.maxContext = std::min(relaunchContext, planContext);
  } else if (!config.nativeLoop.engine.maxContext) {
    // The automatic context, which the plan holds and every start of the
    // model on this Mac that takes its calibration serves alike
    // (AneFfnOutcome::context).
    config.nativeLoop.engine.maxContext = resources->aneFfnOutcome().context;
  } else if (config.nativeLoop.engine.maxContext > planContext) {
    // --max-memory sets the budget only below this Mac's own; the SSD cache's
    // room bounds a context whose KV spills (ContextBounds::bound).
    const auto &budget = resources->memoryPlan().breakdown();
    const bool memoryCapped = budget.configuredMemoryLimitBytes &&
                              budget.hardBudgetBytes == budget.configuredMemoryLimitBytes;
    fail(std::move(base), RuntimeBootstrapStage::ModelCreation,
         "--max-context " + std::to_string(config.nativeLoop.engine.maxContext) +
             " exceeds the " + std::to_string(planContext) + " tokens the model and " +
             (budget.contextBounds.bound() == "disk" ? "the SSD cache's room for its KV"
              : memoryCapped                         ? "--max-memory"
                                                     : "this Mac's memory") +
             " allow; omit it or pass at most " + std::to_string(planContext));
  }
  config.nativeLoop.engine.vocabularySize =
      config.resources.model.capabilities.vocabularySize;
  // A prefill command takes no more attention than a full chunk at the end
  // of the native window, which every supported Mac runs within the command
  // watchdog: past it, at depths only YaRN serves, commands take fewer rows.
  // A row of a target with an indexer attends to its top blocks at any depth,
  // and only the indexer's scores grow with it, a hundredth of attention's,
  // so its commands keep their rows.
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  config.nativeLoop.engine.maximumPrefillAttentionPairs = std::visit(
      [](const auto &layout) {
        return layout.hasIndexer() ? std::numeric_limits<uint64_t>::max()
                                   : Scheduler::attentionPairs(layout.nativeContextTokens - chunk, chunk);
      },
      config.resources.model.target);
  // Images are admitted up to the server's pixel cap; a model without vision
  // admits none.
  config.nativeLoop.engine.maxImagePatches =
      config.resources.model.hasVision() ? config.resources.maximumImagePatches
                                         : 0;

  std::unique_ptr<model::RuntimeModel> modelRuntime;
  try {
    // The plan's fixed bytes already proved the arenas' sum fits.
    const auto &budget = resources->memoryPlan().breakdown();
    // Admission reports a refusal by its failure alone. One raised while the
    // runtime allocates keeps its own message, which says what to do.
    std::string refusal;
    const metal::AllocationResult arenas =
        resources->memoryGovernor().allocationAdmission()(
            budget.sharedPrefillBytes + budget.sharedDecodeBytes, [&] {
              try {
                modelRuntime = model::createRuntime(resources->modelContext());
              } catch (const metal::MetalAllocationError &error) {
                refusal = error.what();
                throw;
              }
            });
    if (!arenas) {
      throw metal::MetalAllocationError(
          refusal.empty() ? std::string("unable to admit model arenas: ") +
                                metal::allocationFailureName(arenas.failure)
                          : refusal,
          arenas.failure);
    }
  } catch (const metal::MetalAllocationError &error) {
    base.resourceFailure = resourceAllocationFailure(error.failure());
    fail(std::move(base), RuntimeBootstrapStage::ModelCreation,
         std::string("modelRuntime creation failed: ") + error.what());
  } catch (const std::exception &error) {
    fail(std::move(base), RuntimeBootstrapStage::ModelCreation,
         std::string("modelRuntime creation failed: ") + error.what());
  } catch (...) {
    fail(std::move(base), RuntimeBootstrapStage::ModelCreation,
         "modelRuntime creation failed with an unknown exception");
  }
  std::unique_ptr<NativeRuntime> nativeLoop;
  try {
    connectToGovernor(config.nativeLoop.engine, resources->memoryGovernor());
    config.nativeLoop.engine.spill = resources->spillConfig();
    config.nativeLoop.weights = &resources->releasableMemory();
    config.nativeLoop.weightAdmission = resources->memoryGovernor().allocationAdmission();
    config.nativeLoop.engineFailed = [owner = resources.get()] { owner->engineFailed(); };
    // The parser and engine consume the same resolved ceiling. In automatic
    // mode it cannot be known until resource planning has measured the device.
    nativeLoop = std::make_unique<NativeRuntime>(
        config.nativeLoop, config.resources.idleReleaseSeconds, resources->cache(),
        *modelRuntime, std::move(output), std::move(statusProvider),
        protocolLimitsFor(config.resources.model.capabilities,
                          config.nativeLoop.engine.maxContext));
  } catch (const metal::MetalAllocationError &error) {
    base.resourceFailure = resourceAllocationFailure(error.failure());
    fail(std::move(base), RuntimeBootstrapStage::ModelCreation,
         std::string("native loop creation failed: ") + error.what());
  } catch (const std::exception &error) {
    fail(std::move(base), RuntimeBootstrapStage::ModelCreation,
         std::string("native loop creation failed: ") + error.what());
  }

  RuntimeResources *resourcesPointer = resources.get();
  model::RuntimeModel *modelPointer = modelRuntime.get();
  RuntimeBootstrapReport report = requireWarmupAndAnnounce(
      resources->memoryPlan(), *modelRuntime,
      [resourcesPointer, modelPointer] {
        // Audit every attempted warmup before reclaiming idle buffers.
        // Wider batches and cache memory grow on demand after Ready.
        ActualMemoryReport report = resourcesPointer->actualMemoryReport(
            modelPointer->actualRuntimeMemory());
        // Keep what the first request starts from: one lane's state buffers
        // and one empty KV extent. No cache data is evicted.
        while (modelPointer->reclaimIdleState(true, model::IdleMemory::Buffers)) {
        }
        static_cast<void>(resourcesPointer->cache().releaseEmptyExtents(true));
        // The last check of the host's reserve before Ready weighs what the
        // engine serves with, not the lanes the optional warmups left idle.
        const auto deadline = AwakeClock::now() + kMemoryReleaseWait;
        for (;;) {
          try {
            resourcesPointer->backend().checkOperation();
            return report;
          } catch (const metal::MetalAllocationError &) {
            if (AwakeClock::now() >= deadline)
              throw;
          }
          std::this_thread::sleep_for(kMemoryReleasePoll);
        }
      },
      *nativeLoop);

  // The per-operation guard RuntimeResources installed is only for startup:
  // once Ready, the engine meets memory pressure between its ticks.
  resources->backend().setOperationGuard({});
  resources->beginServing();
  return std::unique_ptr<RuntimeBootstrap>(
      new RuntimeBootstrap(std::move(resources), std::move(modelRuntime),
                           std::move(nativeLoop), std::move(report)));
}

} // namespace splash::engine
