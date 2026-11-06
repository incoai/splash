#pragma once

#include "engine/AneFfnStartup.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryAudit.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/RuntimeMetrics.hpp"
#include "engine/ThermalState.hpp"
#include "engine/WeightResidency.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ExpertStoreStats.hpp"
#include "model/KvReadStats.hpp"
#include "model/Model.hpp"
#include "model/SparseRowStore.hpp"

#include <array>
#include <optional>
#include <string>

namespace splash::engine {

struct RuntimeCacheIdentity;

enum class WarmupStepStatus { Pending, Complete, MemoryLimited };

struct WarmupReport {
  WarmupStepStatus maximumPrefill = WarmupStepStatus::Pending;
  std::array<WarmupStepStatus, model::ExecutionLimits::maximumBatchWidth>
      decodeBatches{};
  WarmupStepStatus compositeStateRestore = WarmupStepStatus::Pending;
  // Exact executor-selected kernel geometry for the fixed 2048-row path.
  std::string maximumPrefillDetail;
};

// What the start fixed, which every status document reports: held by
// reference, so a StatusStart lives no longer than the values it names.
struct StatusStart final {
  const EngineMemoryPlan &plan;
  const RuntimeCacheIdentity &cacheIdentity;
  const WarmupReport &warmup;
  const MemoryAuditResult &memoryAudit;
};

// One sample of the serving runtime, taken on the loop thread.
struct StatusSample final {
  // A spill-capable target's KV on the SSD cache (kv_spill): what the cache
  // holds and did, and its readers' memory and reads.
  struct KvSpill final {
    KvSpillSnapshot cache;
    std::optional<model::KvBlockCacheStats> blockCache;
    std::optional<model::KvHistoryStreamStats> historyStream;
  };

  EngineSnapshot engine;
  ResourceWaitSnapshot resourceWait;
  WeightsSnapshot weights;
  RuntimeMetricsSnapshot metrics;
  model::ModelTelemetry telemetry;
  metal::MetalMemoryStats metalMemory;
  bool metalHealthy = true;
  // Why Metal failed; empty while it is healthy.
  std::string metalFailureReason;
  MemoryGovernorSnapshot memoryGovernor;
  AneFfnSnapshot aneFfn;
  ThermalState thermalState = ThermalState::Nominal;
  // The loop's longest tick, which its transport measures.
  double loopMaxTickMilliseconds = 0.0;
  std::optional<model::SparseRowStats> streamedTable;
  std::optional<model::ExpertStoreStats> streamedExperts;
  // None for a target that keeps its KV in memory.
  std::optional<KvSpill> kvSpill;
};

// Single source for /status and native protocol status events.
[[nodiscard]] std::string runtimeStatusJson(const StatusStart &start, const StatusSample &sample);

} // namespace splash::engine
