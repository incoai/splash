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

// A spill-capable target's plan (KvSpillPlan), as /status reports it; null
// for a target that keeps its KV in memory.
void writeSpillPlan(json::Writer &w, const KvSpillPlan &spill) {
  if (!spill.enabled) {
    w.null();
    return;
  }
  w.object([&] {
    w.field("dense_pages", spill.densePages)
        .field("window_pages", spill.windowPages)
        .field("group_pages", spill.groupPages)
        .field("slot_bytes", spill.slotBytes)
        .field("lane_slots", spill.laneSlots)
        .field("hot_floor_bytes", spill.hotFloorBytes)
        .field("hot_target_bytes", spill.hotTargetBytes)
        .field("hot_limit_bytes", spill.hotLimitBytes)
        .field("staging_buffers", spill.stagingBuffers)
        .field("staging_bytes", spill.stagingBytes)
        .field("index_runway_bytes", spill.indexRunwayBytes)
        .field("floor_tokens", spill.floorTokens)
        .field("floor_bytes", spill.floorBytes);
  });
}

void writeDevice(json::Writer &w, const DeviceCapabilities &device) {
  w.object([&] {
    w.field("device_name", device.deviceName)
        .field("macos_version", device.macosVersion())
        .field("apple_gpu_family", device.appleGpuFamily)
        .field("gpu_core_count", device.gpuCoreCount)
        .field("physical_memory_bytes", device.physicalMemoryBytes)
        .field("recommended_max_working_set_bytes", device.recommendedMaxWorkingSetBytes)
        .field("max_buffer_length_bytes", device.maxBufferLengthBytes)
        .field("max_threadgroup_memory_bytes", device.maxThreadgroupMemoryBytes)
        .field("max_threadgroup_width", device.maxThreadgroupWidth)
        .field("has_unified_memory", device.hasUnifiedMemory);
  });
}

void writeModel(json::Writer &w, const ModelMemoryProfile &model) {
  const ModelMemoryFootprint &footprint = model.footprint;
  w.object([&] {
    w.field("model_name", model.name)
        .field("maximum_context_tokens", model.maximumContextTokens)
        .field("attention_layers", model.targetKvLayout.attentionLayers)
        .field("kv_heads", model.targetKvLayout.kvHeads)
        .field("head_dimension", model.targetKvLayout.headDimension)
        .field("kv_page_tokens", kv::kPageTokens)
        .field("kv_format", kv::formatName(model.targetKvLayout.format))
        .field("kv_elements_per_scale", model.targetKvLayout.elementsPerScale())
        .field("kv_page_bytes", model.targetKvLayout.bytesPerModelPage())
        .field("state_format", ops::formatName(model.stateFormat));
    w.object("memory", [&] {
      w.field("target_weights_bytes", footprint.targetWeightsBytes)
          .field("draft_weights_bytes", footprint.draftWeightsBytes)
          .field("vision_weights_bytes", footprint.visionWeightsBytes)
          .field("lane_state_bytes", footprint.runtime.laneStatePlannedAllocatedBytes)
          .field("shared_prefill_bytes", footprint.runtime.sharedPrefillPlannedAllocatedBytes)
          .field("shared_decode_bytes", footprint.runtime.sharedDecodePlannedAllocatedBytes)
          .field("pipeline_reserve_bytes", model::kPipelineReserveBytes)
          .field("runtime_overhead_reserve_bytes", model::kRuntimeOverheadReserveBytes)
          .field("streamed_table_bytes", footprint.streamedTableBytes)
          .field("ane_ffn_bytes", footprint.aneFfnBytes)
          .field("state_staging_bytes", footprint.stateStagingBytes);
    });
  });
}

void writeBudget(json::Writer &w, const EngineMemoryBreakdown &budget) {
  w.object([&] {
    w.field("physical_memory_bytes", budget.physicalMemoryBytes)
        .field("recommended_working_set_bytes", budget.recommendedWorkingSetBytes)
        .field("configured_memory_limit_bytes", budget.configuredMemoryLimitBytes)
        .field("working_set_margin_bytes", budget.workingSetMarginBytes)
        .field("hard_budget_bytes", budget.hardBudgetBytes)
        .field("target_weights_bytes", budget.targetWeightsBytes)
        .field("draft_weights_bytes", budget.draftWeightsBytes)
        .field("vision_weights_bytes", budget.visionWeightsBytes)
        .field("maximum_batch_width", budget.maximumBatchWidth)
        .field("lane_state_bytes", budget.laneStateBytes)
        .field("shared_prefill_bytes", budget.sharedPrefillBytes)
        .field("shared_decode_bytes", budget.sharedDecodeBytes)
        .field("pipeline_reserve_bytes", budget.pipelineReserveBytes)
        .field("runtime_overhead_reserve_bytes", budget.runtimeOverheadReserveBytes)
        .field("streamed_table_bytes", budget.streamedTableBytes)
        .field("ane_ffn_bytes", budget.aneFfnBytes)
        .field("state_staging_bytes", budget.stateStagingBytes)
        .field("host_held_back_bytes", budget.hostHeldBackBytes)
        .field("fixed_runtime_bytes", budget.fixedRuntimeBytes)
        .field("dynamic_budget_bytes", budget.dynamicBudgetBytes)
        .field("kv_page_tokens", budget.kvPageTokens)
        .field("kv_page_bytes", budget.kvPageBytes)
        .field("kv_extent_pages", budget.kvExtentPages)
        .field("kv_extent_bytes", budget.kvExtentBytes)
        .field("kv_capacity_pages", budget.kvCapacityPages)
        .field("kv_capacity_bytes", budget.kvCapacityBytes)
        .field("kv_capacity_tokens", budget.kvCapacityTokens)
        .field("kv_index_page_bytes", budget.kvIndexPageBytes);
    // A windowed layout's Slim pages; nothing for another layout.
    if (budget.kvSlimPageBytes) {
      w.object("kv_window", [&] {
        w.field("slim_page_bytes", budget.kvSlimPageBytes).field("floor_pages", budget.kvWindowFloorPages);
      });
    }
    w.key("kv_spill");
    writeSpillPlan(w, budget.kvSpill);
    w.object("context_bounds", [&] {
      const ContextBounds &bounds = budget.contextBounds;
      w.field("window", bounds.window)
          .field("resident", bounds.resident)
          .field("spilled", bounds.spilled)
          .field("disk", bounds.disk)
          .field("bound", bounds.bound());
    });
    w.field("minimum_dynamic_bytes", budget.minimumDynamicBytes)
        .field("minimum_required_bytes", budget.minimumRequiredBytes)
        .field("deficit_bytes", budget.deficitBytes);
  });
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

uint64_t kvCapacityPagesFor(const kv::Layout &layout, uint64_t pages) noexcept {
  const uint32_t extentPages = layout.extentPagesFor(pages);
  return extentPages ? pages - pages % extentPages : 0;
}

uint32_t kvWindowFloorPages(const kv::Layout &layout, uint32_t commandPages) noexcept {
  if (!layout.slimmable()) return 0;
  const uint32_t command =
      commandPages ? commandPages : model::ExecutionLimits::prefillTokenBudget / kv::kPageTokens;
  return layout.windowPages() + command + 1;
}

uint64_t kvPagesBytes(const kv::Layout &layout, uint64_t pages, uint32_t floorPages, uint32_t extentPages) noexcept {
  const uint64_t full = layout.slimmable() ? std::min<uint64_t>(pages, floorPages) : pages;
  const auto extents = [&](uint64_t count) { return count ? (count - 1) / extentPages + 1 : 0; };
  return extentPages * (extents(full) * layout.bytesPerModelPage() +
                        extents(pages - full) * layout.bytesPerModelPage(kv::PageForm::Slim));
}

uint64_t kvPagesWithin(const kv::Layout &layout, uint64_t bytes, uint32_t floorPages, uint32_t extentPages) noexcept {
  const auto whole = [&](uint64_t room, kv::PageForm form) {
    return room / (extentPages * layout.bytesPerModelPage(form)) * extentPages;
  };
  const uint64_t floorBytes = kvPagesBytes(layout, floorPages, floorPages, extentPages);
  if (!layout.slimmable() || bytes < floorBytes) return whole(bytes, kv::PageForm::Full);
  return floorPages + whole(bytes - floorBytes, kv::PageForm::Slim);
}

uint32_t ContextBounds::maximum() const noexcept {
  return std::min(window, std::max(resident, std::min(spilled, disk)));
}

std::string_view ContextBounds::bound() const noexcept {
  const uint32_t context = maximum();
  if (context == window) return "model";
  return context == resident || disk >= spilled ? "memory" : "disk";
}

std::optional<std::string> ModelMemoryProfile::validationError() const {
  if (name.empty()) return "model_name_required";
  if (!maximumContextTokens ||
      maximumContextTokens > kv::kMaximumLogicalTokens) {
    return "invalid_model_context_length";
  }
  if (!targetKvLayout.valid()) return "invalid_target_kv_layout";
  if (spill.enabled && !targetKvLayout.spillable()) return "spill_plan_needs_a_spillable_layout";
  if (!ops::validFormat(stateFormat)) return "invalid_state_format";
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
           footprint.streamedTableBytes, footprint.aneFfnBytes, footprint.stateStagingBytes,
           footprint.hostHeldBackBytes}) {
    if (!checkedAdd(result, value, result)) {
      throw std::overflow_error("fixed runtime cost overflow");
    }
  }
  return result;
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
      << "streamed table cache: " << bytesAndMiB(streamedTableBytes) << '\n'
      << "Neural Engine split: " << bytesAndMiB(aneFfnBytes) << '\n'
      << "SSD cache state staging: " << bytesAndMiB(stateStagingBytes) << '\n'
      << "expert store above its floor at the start: " << bytesAndMiB(hostHeldBackBytes) << '\n'
      << "fixed runtime: " << bytesAndMiB(fixedRuntimeBytes) << '\n'
      << "elastic state/KV budget: " << bytesAndMiB(dynamicBudgetBytes) << '\n'
      << "KV page: " << kvPageTokens << " tokens, "
      << bytesAndMiB(kvPageBytes) << '\n'
      << "KV extent: " << kvExtentPages << " pages, "
      << bytesAndMiB(kvExtentBytes) << '\n'
      << "KV capacity of one request: " << kvCapacityPages << " pages / "
      << kvCapacityTokens << " tokens\n";
  if (kvSlimPageBytes)
    out << "KV past the windows: Slim pages of " << bytesAndMiB(kvSlimPageBytes) << ", a lane keeping "
        << kvWindowFloorPages << " pages Full\n";
  if (kvSpill.enabled)
    out << "KV on the SSD cache: Index pages of " << bytesAndMiB(kvIndexPageBytes) << "; a Spill request keeps "
        << kvSpill.densePages << " dense, " << kvSpill.windowPages << " window and 2 x " << kvSpill.groupPages
        << " group pages Full; floors of " << bytesAndMiB(kvSpill.floorBytes) << " at "
        << kvSpill.floorTokens << " tokens: hot block cache " << bytesAndMiB(kvSpill.hotFloorBytes)
        << " (target " << bytesAndMiB(kvSpill.hotTargetBytes) << "), history staging "
        << bytesAndMiB(kvSpill.stagingBytes) << " in " << kvSpill.stagingBuffers << ", Index runway "
        << bytesAndMiB(kvSpill.indexRunwayBytes) << '\n';
  out << "context: " << contextBounds.maximum() << " tokens (" << contextBounds.bound() << "; window "
      << contextBounds.window << ", resident " << contextBounds.resident << ", spilled " << contextBounds.spilled
      << ", disk " << contextBounds.disk << ")\n"
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

uint32_t EngineMemoryPlan::maximumContextTokens() const noexcept { return breakdown_.contextBounds.maximum(); }

SpillGeometry EngineMemoryPlan::spillGeometry() const noexcept {
  const KvSpillPlan &spill = breakdown_.kvSpill;
  return {.fullPageBytes = breakdown_.kvPageBytes,
          .indexPageBytes = breakdown_.kvIndexPageBytes,
          .slotBytes = spill.slotBytes,
          .residentTokens = breakdown_.contextBounds.resident,
          .densePages = spill.densePages,
          .windowPages = spill.windowPages,
          .groupPages = spill.groupPages};
}

void writeJson(json::Writer &w, const EngineMemoryPlan &plan) {
  w.object([&] {
    w.field("valid", true).field("maximum_context_tokens", plan.maximumContextTokens()).key("device");
    writeDevice(w, plan.device_);
    w.key("model");
    writeModel(w, plan.model_);
    w.key("budget");
    writeBudget(w, plan.breakdown_);
  });
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
  breakdown.aneFfnBytes = model.footprint.aneFfnBytes;
  breakdown.streamedTableBytes = model.footprint.streamedTableBytes;
  breakdown.hostHeldBackBytes = model.footprint.hostHeldBackBytes;
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
  breakdown.kvIndexPageBytes = model.targetKvLayout.indexBytesPerModelPage();
  breakdown.kvSpill = model.spill;
  // A windowed layout's request keeps Full only its lane's window and the
  // pages its command writes; the rest of its KV is Slim.
  const kv::Layout &kvLayout = model.targetKvLayout;
  breakdown.kvWindowFloorPages = kvWindowFloorPages(kvLayout, model.commandPages);
  if (kvLayout.slimmable())
    breakdown.kvSlimPageBytes = kvLayout.bytesPerModelPage(kv::PageForm::Slim);
  // Page ids stay 32-bit. One request's KV capacity is the whole extents of
  // the size that leaves the fewest of the budget's pages, priced one by one,
  // unused; a windowed layout's Full and Slim pages take extents of their
  // own (kvPagesWithin).
  const uint64_t availableForOneRequestKv =
      breakdown.dynamicBudgetBytes > breakdown.laneStateBytes
          ? breakdown.dynamicBudgetBytes - breakdown.laneStateBytes
          : 0;
  const auto pagesWithin = [&](uint32_t extentPages) {
    return std::min<uint64_t>(
        kvPagesWithin(kvLayout, availableForOneRequestKv, breakdown.kvWindowFloorPages, extentPages),
        std::numeric_limits<uint32_t>::max());
  };
  const uint64_t budgetPages = pagesWithin(1);
  breakdown.kvExtentPages = model.targetKvLayout.extentPagesFor(budgetPages);
  breakdown.kvCapacityPages =
      static_cast<uint32_t>(kvLayout.slimmable() && breakdown.kvExtentPages
                                ? pagesWithin(breakdown.kvExtentPages)
                                : kvCapacityPagesFor(model.targetKvLayout, budgetPages));
  breakdown.kvCapacityBytes =
      kvPagesBytes(kvLayout, breakdown.kvCapacityPages, breakdown.kvWindowFloorPages, breakdown.kvExtentPages);

  if (!checkedMultiply(breakdown.kvPageBytes, breakdown.kvExtentPages,
                       breakdown.kvExtentBytes) ||
      !checkedMultiply(breakdown.kvPageTokens, breakdown.kvCapacityPages,
                       breakdown.kvCapacityTokens)) {
    return {std::nullopt, failure(BudgetErrorCode::ArithmeticOverflow,
                                  "KV capacity overflows uint64",
                                  std::move(breakdown))};
  }
  const KvSpillPlan &spill = model.spill;
  breakdown.contextBounds = {
      .window = model.maximumContextTokens,
      .resident = static_cast<uint32_t>(std::min<uint64_t>(
          model.maximumContextTokens,
          saturatingSubtract(breakdown.kvCapacityTokens, model::ExecutionLimits::speculativeScratchTokens))),
      .spilled = spill.memoryTokens,
      .disk = spill.diskTokens};
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
