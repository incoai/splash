#include "engine/MemoryPlan.hpp"
#include "Checked.hpp"
#include "engine/Json.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::engine {
namespace {

std::string bytesAndMiB(uint64_t bytes) {
  std::ostringstream out;
  out << bytes << " bytes (" << std::fixed << std::setprecision(2)
      << static_cast<long double>(bytes) / kMiB << " MiB)";
  return out.str();
}

BudgetValidationStatus failure(BudgetErrorCode code, std::string message,
                               EngineMemoryBreakdown breakdown) {
  return {false, code, std::move(message), std::move(breakdown)};
}

std::string_view budgetErrorCodeName(BudgetErrorCode code) {
  switch (code) {
  case BudgetErrorCode::None:
    return "none";
  case BudgetErrorCode::InvalidDeviceCapabilities:
    return "invalid_device_capabilities";
  case BudgetErrorCode::InvalidModelSpec:
    return "invalid_model_spec";
  case BudgetErrorCode::WorkingSetTooSmall:
    return "working_set_too_small";
  case BudgetErrorCode::ArithmeticOverflow:
    return "arithmetic_overflow";
  case BudgetErrorCode::KvPoolDoesNotFit:
    return "kv_pool_does_not_fit";
  }
  return "unknown";
}

std::string deviceStatusJson(const DeviceCapabilities &device) {
  std::ostringstream out;
  out << '{' << "\"device_name\":" << json::quote(device.deviceName) << ','
      << "\"macos_version\":" << json::quote(device.macosVersion()) << ','
      << "\"apple_gpu_family\":" << device.appleGpuFamily << ','
      << "\"gpu_core_count\":" << device.gpuCoreCount << ','
      << "\"physical_memory_bytes\":" << device.physicalMemoryBytes << ','
      << "\"recommended_max_working_set_bytes\":"
      << device.recommendedMaxWorkingSetBytes << ','
      << "\"max_buffer_length_bytes\":" << device.maxBufferLengthBytes << ','
      << "\"max_threadgroup_memory_bytes\":"
      << device.maxThreadgroupMemoryBytes << ','
      << "\"max_threadgroup_width\":" << device.maxThreadgroupWidth << ','
      << "\"has_unified_memory\":"
      << (device.hasUnifiedMemory ? "true" : "false") << '}';
  return out.str();
}

std::string modelStatusJson(const ModelMemoryProfile &model) {
  std::ostringstream out;
  out << '{' << "\"model_name\":" << json::quote(model.name) << ','
      << "\"maximum_context_tokens\":" << model.maximumContextTokens << ','
      << "\"attention_layers\":" << model.targetKvLayout.attentionLayers << ','
      << "\"kv_heads\":" << model.targetKvLayout.kvHeads << ','
      << "\"head_dimension\":" << model.targetKvLayout.headDimension << ','
      << "\"kv_page_tokens\":" << kv::kPageTokens << ','
      << "\"kv_format\":" << json::quote(kv::formatName(model.targetKvLayout.format)) << ','
      << "\"kv_elements_per_scale\":"
      << model.targetKvLayout.elementsPerScale() << ','
      << "\"kv_page_bytes\":" << model.targetKvLayout.bytesPerModelPage() << ','
      << "\"memory\":{" << "\"target_weights_bytes\":"
      << model.footprint.targetWeightsBytes << ','
      << "\"draft_weights_bytes\":" << model.footprint.draftWeightsBytes << ','
      << "\"vision_weights_bytes\":" << model.footprint.visionWeightsBytes << ','
      << "\"lane_state_bytes\":"
      << model.footprint.runtime.laneStatePlannedAllocatedBytes << ','
      << "\"shared_prefill_bytes\":"
      << model.footprint.runtime.sharedPrefillPlannedAllocatedBytes << ','
      << "\"shared_decode_bytes\":"
      << model.footprint.runtime.sharedDecodePlannedAllocatedBytes << ','
      << "\"pipeline_reserve_bytes\":" << model::kPipelineReserveBytes << ','
      << "\"runtime_overhead_reserve_bytes\":"
      << model::kRuntimeOverheadReserveBytes << ','
      << "\"state_staging_bytes\":" << model.footprint.stateStagingBytes << "}}";
  return out.str();
}

} // namespace

std::optional<uint64_t> minimumRequiredBytes(uint64_t fixedBytes,
                                             uint64_t laneStateBytes,
                                             const kv::Layout &layout) noexcept {
  uint64_t runwayBytes = 0;
  uint64_t dynamicBytes = 0;
  uint64_t result = 0;
  if (!checkedMultiply(layout.bytesPerModelPage(),
                       kvRunwayPages(layout.minimumExtentPages()), runwayBytes) ||
      !checkedAdd(laneStateBytes, runwayBytes, dynamicBytes) ||
      !checkedAdd(fixedBytes, dynamicBytes, result))
    return std::nullopt;
  return result;
}

std::optional<std::string> ModelMemoryProfile::validationError() const {
  if (name.empty()) return "model_name_required";
  if (!maximumContextTokens ||
      maximumContextTokens > kv::kMaximumLogicalTokens) {
    return "invalid_model_context_length";
  }
  if (!targetKvLayout.valid()) return "invalid_target_kv_layout";
  if (!footprint.targetWeightsBytes) return "target_weight_bytes_required";
  if (!footprint.draftWeightsBytes) return "draft_weight_bytes_required";
  if (!footprint.runtime.laneStatePlannedAllocatedBytes) {
    return "lane_state_bytes_required";
  }
  if (!footprint.runtime.sharedPrefillPlannedAllocatedBytes) {
    return "shared_prefill_bytes_required";
  }
  if (!footprint.runtime.sharedDecodePlannedAllocatedBytes) {
    return "shared_decode_bytes_required";
  }
  try {
    static_cast<void>(fixedRuntimeBytes());
  } catch (const std::overflow_error &) {
    return "fixed_runtime_cost_overflow";
  }
  return std::nullopt;
}

uint64_t ModelMemoryProfile::fixedRuntimeBytes() const {
  uint64_t result = 0;
  for (uint64_t value : {
           footprint.targetWeightsBytes, footprint.draftWeightsBytes,
           footprint.visionWeightsBytes,
           footprint.runtime.sharedPrefillPlannedAllocatedBytes,
           footprint.runtime.sharedDecodePlannedAllocatedBytes,
           model::kPipelineReserveBytes, model::kRuntimeOverheadReserveBytes,
           footprint.stateStagingBytes}) {
    if (!checkedAdd(result, value, result)) {
      throw std::overflow_error("fixed runtime cost overflow");
    }
  }
  return result;
}

std::string EngineMemoryBreakdown::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"physical_memory_bytes\":" << physicalMemoryBytes << ','
      << "\"recommended_working_set_bytes\":" << recommendedWorkingSetBytes
      << ','
      << "\"configured_memory_limit_bytes\":" << configuredMemoryLimitBytes
      << ',' << "\"working_set_margin_bytes\":" << workingSetMarginBytes << ','
      << "\"hard_budget_bytes\":" << hardBudgetBytes << ','
      << "\"target_weights_bytes\":" << targetWeightsBytes << ','
      << "\"draft_weights_bytes\":" << draftWeightsBytes << ','
      << "\"vision_weights_bytes\":" << visionWeightsBytes << ','
      << "\"maximum_batch_width\":" << maximumBatchWidth << ','
      << "\"lane_state_bytes\":" << laneStateBytes << ','
      << "\"shared_prefill_bytes\":" << sharedPrefillBytes << ','
      << "\"shared_decode_bytes\":" << sharedDecodeBytes << ','
      << "\"pipeline_reserve_bytes\":" << pipelineReserveBytes << ','
      << "\"runtime_overhead_reserve_bytes\":" << runtimeOverheadReserveBytes
      << ',' << "\"state_staging_bytes\":" << stateStagingBytes << ','
      << "\"fixed_runtime_bytes\":" << fixedRuntimeBytes << ','
      << "\"dynamic_budget_bytes\":" << dynamicBudgetBytes << ','
      << "\"kv_page_tokens\":" << kvPageTokens << ','
      << "\"kv_page_bytes\":" << kvPageBytes << ','
      << "\"kv_extent_pages\":" << kvExtentPages << ','
      << "\"kv_extent_bytes\":" << kvExtentBytes << ','
      << "\"kv_capacity_pages\":" << kvCapacityPages << ','
      << "\"kv_capacity_bytes\":" << kvCapacityBytes << ','
      << "\"kv_capacity_tokens\":" << kvCapacityTokens << ','
      << "\"minimum_dynamic_bytes\":" << minimumDynamicBytes << ','
      << "\"minimum_required_bytes\":" << minimumRequiredBytes << ','
      << "\"deficit_bytes\":" << deficitBytes << '}';
  return out.str();
}

std::string EngineMemoryBreakdown::describe() const {
  std::ostringstream out;
  out << "physical memory: " << bytesAndMiB(physicalMemoryBytes) << '\n'
      << "recommended working set: " << bytesAndMiB(recommendedWorkingSetBytes)
      << '\n'
      << "configured memory limit: "
      << (configuredMemoryLimitBytes ? bytesAndMiB(configuredMemoryLimitBytes)
                                     : "automatic")
      << '\n'
      << "working-set margin (max of 1 GiB or 2%): "
      << bytesAndMiB(workingSetMarginBytes) << '\n'
      << "hard budget: " << bytesAndMiB(hardBudgetBytes) << '\n'
      << "target weights: " << bytesAndMiB(targetWeightsBytes) << '\n'
      << "draft weights: " << bytesAndMiB(draftWeightsBytes) << '\n'
      << "vision weights: " << bytesAndMiB(visionWeightsBytes) << '\n'
      << "maximum DFlash batch width: " << maximumBatchWidth << '\n'
      << "lane state: " << bytesAndMiB(laneStateBytes) << '\n'
      << "shared prefill: " << bytesAndMiB(sharedPrefillBytes) << '\n'
      << "shared decode: " << bytesAndMiB(sharedDecodeBytes) << '\n'
      << "pipeline reserve: " << bytesAndMiB(pipelineReserveBytes) << '\n'
      << "allocator/runtime reserve: "
      << bytesAndMiB(runtimeOverheadReserveBytes) << '\n'
      << "SSD cache state staging: " << bytesAndMiB(stateStagingBytes) << '\n'
      << "fixed runtime: " << bytesAndMiB(fixedRuntimeBytes) << '\n'
      << "elastic state/KV budget: " << bytesAndMiB(dynamicBudgetBytes) << '\n'
      << "KV page: " << kvPageTokens << " tokens, "
      << bytesAndMiB(kvPageBytes) << '\n'
      << "KV extent: " << kvExtentPages << " pages, "
      << bytesAndMiB(kvExtentBytes) << '\n'
      << "KV capacity of one request: " << kvCapacityPages << " pages / "
      << kvCapacityTokens << " tokens\n"
      << "minimum dynamic runtime: " << bytesAndMiB(minimumDynamicBytes) << '\n'
      << "minimum required: " << bytesAndMiB(minimumRequiredBytes) << '\n'
      << "deficit: " << bytesAndMiB(deficitBytes);
  return out.str();
}

std::string BudgetValidationStatus::describe() const {
  std::ostringstream out;
  out << (valid ? "engine memory plan valid"
                : "engine memory budget validation failed")
      << " [" << budgetErrorCodeName(code) << "]: " << message << '\n'
      << breakdown.describe();
  return out.str();
}

EngineMemoryPlan::EngineMemoryPlan(DeviceCapabilities device,
                                   ModelMemoryProfile model,
                                   EngineMemoryBreakdown breakdown)
    : device_(std::move(device)), model_(std::move(model)),
      breakdown_(std::move(breakdown)) {}

uint32_t EngineMemoryPlan::maximumContextTokens() const noexcept {
  const uint64_t kvTokens = breakdown_.kvCapacityTokens;
  const uint64_t contextTokens =
      kvTokens > model::ExecutionLimits::speculativeScratchTokens
          ? kvTokens - model::ExecutionLimits::speculativeScratchTokens
          : 0;
  return static_cast<uint32_t>(
      std::min<uint64_t>(model_.maximumContextTokens, contextTokens));
}

uint32_t EngineMemoryPlan::contextTokensWithin(uint64_t memoryBytes) const {
  if (!memoryBytes)
    return 0;
  const uint64_t configured = breakdown_.configuredMemoryLimitBytes;
  const EngineMemoryPlanResult within = evaluateEngineMemoryPlan(
      device_, model_, configured ? std::min(configured, memoryBytes) : memoryBytes);
  return within.plan ? within.plan->maximumContextTokens() : 0;
}

std::string EngineMemoryPlan::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"valid\":true,"
      << "\"maximum_context_tokens\":" << maximumContextTokens() << ','
      << "\"device\":" << deviceStatusJson(device_) << ','
      << "\"model\":" << modelStatusJson(model_) << ','
      << "\"budget\":" << breakdown_.toStatusJson() << '}';
  return out.str();
}

EngineMemoryPlanResult
evaluateEngineMemoryPlan(const DeviceCapabilities &device,
                         const ModelMemoryProfile &model,
                         uint64_t maximumMemoryBytes) {
  EngineMemoryBreakdown breakdown;
  breakdown.physicalMemoryBytes = device.physicalMemoryBytes;
  breakdown.recommendedWorkingSetBytes = device.recommendedMaxWorkingSetBytes;
  breakdown.configuredMemoryLimitBytes = maximumMemoryBytes;
  breakdown.targetWeightsBytes = model.footprint.targetWeightsBytes;
  breakdown.draftWeightsBytes = model.footprint.draftWeightsBytes;
  breakdown.visionWeightsBytes = model.footprint.visionWeightsBytes;
  breakdown.laneStateBytes =
      model.footprint.runtime.laneStatePlannedAllocatedBytes;
  breakdown.sharedPrefillBytes =
      model.footprint.runtime.sharedPrefillPlannedAllocatedBytes;
  breakdown.sharedDecodeBytes =
      model.footprint.runtime.sharedDecodePlannedAllocatedBytes;
  breakdown.pipelineReserveBytes = model::kPipelineReserveBytes;
  breakdown.runtimeOverheadReserveBytes = model::kRuntimeOverheadReserveBytes;
  breakdown.stateStagingBytes = model.footprint.stateStagingBytes;
  breakdown.kvPageTokens = kv::kPageTokens;

  if (auto error = device.validationError()) {
    return {std::nullopt, failure(BudgetErrorCode::InvalidDeviceCapabilities,
                                  *error, std::move(breakdown))};
  }
  if (auto error = model.validationError()) {
    return {std::nullopt, failure(BudgetErrorCode::InvalidModelSpec, *error,
                                  std::move(breakdown))};
  }

  breakdown.workingSetMarginBytes = EngineMemoryPolicy::workingSetMarginBytes(
      breakdown.recommendedWorkingSetBytes);
  if (breakdown.recommendedWorkingSetBytes <=
      breakdown.workingSetMarginBytes) {
    return {std::nullopt,
            failure(BudgetErrorCode::WorkingSetTooSmall,
                    "recommended working set does not exceed the working-set "
                    "margin",
                    std::move(breakdown))};
  }
  breakdown.hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
      breakdown.recommendedWorkingSetBytes, maximumMemoryBytes);

  breakdown.kvPageBytes = model.targetKvLayout.bytesPerModelPage();
  breakdown.fixedRuntimeBytes = model.fixedRuntimeBytes();
  breakdown.dynamicBudgetBytes =
      breakdown.hardBudgetBytes > breakdown.fixedRuntimeBytes
          ? breakdown.hardBudgetBytes - breakdown.fixedRuntimeBytes
          : 0;
  const std::optional<uint64_t> required =
      minimumRequiredBytes(breakdown.fixedRuntimeBytes,
                           breakdown.laneStateBytes, model.targetKvLayout);
  if (!required) {
    return {std::nullopt,
            failure(BudgetErrorCode::ArithmeticOverflow,
                    "minimum elastic runtime footprint overflows uint64",
                    std::move(breakdown))};
  }
  breakdown.minimumRequiredBytes = *required;
  breakdown.minimumDynamicBytes = *required - breakdown.fixedRuntimeBytes;
  // Page ids stay 32-bit. One request's KV capacity is the whole extents of
  // the size that leaves the fewest of the budget's pages unused.
  const uint64_t availableForOneRequestKv =
      breakdown.dynamicBudgetBytes > breakdown.laneStateBytes
          ? breakdown.dynamicBudgetBytes - breakdown.laneStateBytes
          : 0;
  const uint64_t budgetPages =
      std::min<uint64_t>(availableForOneRequestKv / breakdown.kvPageBytes,
                         std::numeric_limits<uint32_t>::max());
  breakdown.kvExtentPages = model.targetKvLayout.extentPagesFor(budgetPages);
  if (breakdown.kvExtentPages) {
    breakdown.kvCapacityPages = static_cast<uint32_t>(
        budgetPages - budgetPages % breakdown.kvExtentPages);
  }

  if (!checkedMultiply(breakdown.kvPageBytes, breakdown.kvExtentPages,
                       breakdown.kvExtentBytes) ||
      !checkedMultiply(breakdown.kvPageBytes, breakdown.kvCapacityPages,
                       breakdown.kvCapacityBytes) ||
      !checkedMultiply(breakdown.kvPageTokens, breakdown.kvCapacityPages,
                       breakdown.kvCapacityTokens)) {
    return {std::nullopt, failure(BudgetErrorCode::ArithmeticOverflow,
                                  "KV capacity overflows uint64",
                                  std::move(breakdown))};
  }
  if (!breakdown.kvExtentPages ||
      breakdown.dynamicBudgetBytes < breakdown.minimumDynamicBytes) {
    if (breakdown.minimumRequiredBytes > breakdown.hardBudgetBytes) {
      breakdown.deficitBytes =
          breakdown.minimumRequiredBytes - breakdown.hardBudgetBytes;
    }
    return {
        std::nullopt,
        failure(
            BudgetErrorCode::KvPoolDoesNotFit,
            "hard budget cannot fit one lane's state and the KV runway",
            std::move(breakdown))};
  }

  BudgetValidationStatus status{true, BudgetErrorCode::None,
                                "memory plan fits hard budget", breakdown};
  EngineMemoryPlan plan(device, model, breakdown);
  return {std::move(plan), std::move(status)};
}

} // namespace splash::engine
