// Per-kernel GPU time attribution for the production executor.
//
//   decode-profile METALLIB MODEL_ROOT [--prompt-tokens N] [--cycles K] [--kv-format int8|bf16]
//
// Drives the real model runtime with Metal dispatch profiling enabled, so
// every dispatch of a packed prefill command and B1 through B4 DFlash cycles
// is replayed as its own command and attributed to its pipeline.
// The fused (unprofiled) GPU time of the same work is reported alongside, so
// the gap between the sum of parts and the fused command shows how much a
// cycle pays in dispatch boundaries rather than kernel work.

#include "engine/Types.hpp"
#include "model/Runtime.hpp"
#include "ops/PageStorage.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

struct Attribution final {
  uint64_t dispatches = 0;
  double gpuSeconds = 0.0;
};

using Table = std::map<std::string, Attribution>;

void accumulate(Table &table, std::span<const metal::DispatchTiming> timings) {
  for (const metal::DispatchTiming &timing : timings) {
    Attribution &entry = table[timing.pipelineName];
    ++entry.dispatches;
    entry.gpuSeconds += timing.gpuSeconds;
  }
}

// `fusedGpuSeconds` is already per unit of work; the table accumulated
// `divisor` units.
void print(const std::string &title, const Table &table, double divisor,
           double fusedGpuSeconds) {
  std::vector<std::pair<std::string, Attribution>> rows(table.begin(),
                                                        table.end());
  std::sort(rows.begin(), rows.end(), [](const auto &left, const auto &right) {
    return left.second.gpuSeconds > right.second.gpuSeconds;
  });
  double total = 0.0;
  uint64_t dispatches = 0;
  for (const auto &[_, entry] : rows) {
    total += entry.gpuSeconds;
    dispatches += entry.dispatches;
  }
  std::printf("\n== %s: %.2f ms fused, %.2f ms as %llu separate dispatches ==\n",
              title.c_str(), fusedGpuSeconds * 1e3, total * 1e3 / divisor,
              static_cast<unsigned long long>(dispatches / divisor));
  std::printf("%-44s %9s %10s %6s\n", "pipeline", "count", "gpu_ms", "share");
  for (const auto &[name, entry] : rows) {
    std::printf("%-44s %9.1f %10.3f %5.1f%%\n", name.c_str(),
                entry.dispatches / divisor, entry.gpuSeconds * 1e3 / divisor,
                entry.gpuSeconds / total * 100.0);
  }
}

std::vector<uint32_t> pageRange(uint32_t first, uint32_t count) {
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index)
    result[index] = first + index;
  return result;
}

// A chat-formatted request that asks for a long answer, so decode cycles keep
// producing tokens instead of a stop token right after prefill. The user text
// repeats one sentence until the prompt reaches the requested length.
std::vector<uint32_t> chatPrompt(uint32_t tokens) {
  static constexpr std::array<uint32_t, 3> kUserHeader{248045, 846, 198};
  static constexpr std::array<uint32_t, 17> kSentence{
      7734, 264, 11346, 11,    7072,  12, 26829, 8627, 883,
      279,  3712, 314,   279,   12386, 19825, 13,    220};
  static constexpr std::array<uint32_t, 9> kAssistantHeader{
      248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
  if (tokens < kUserHeader.size() + kAssistantHeader.size() + 1)
    throw std::invalid_argument("prompt is too short for a chat request");
  std::vector<uint32_t> prompt(kUserHeader.begin(), kUserHeader.end());
  const uint32_t body = tokens - kUserHeader.size() - kAssistantHeader.size();
  for (uint32_t index = 0; index < body; ++index)
    prompt.push_back(kSentence[index % kSentence.size()]);
  prompt.insert(prompt.end(), kAssistantHeader.begin(), kAssistantHeader.end());
  return prompt;
}

uint32_t parseCount(std::string_view text, std::string_view label) {
  uint32_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !value) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return value;
}

struct Lane final {
  uint64_t id = 0;
  uint32_t slot = 0;
  uint64_t position = 0;
  std::vector<uint32_t> pages;
};

void prefill(model::Runtime &executor, Lane &lane,
             std::span<const uint32_t> prompt) {
  EngineRequest request;
  request.id = lane.id;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = 256;
  executor.beginColdRequest(request.modelView(), lane.slot);
  uint32_t offset = 0;
  while (offset < prompt.size()) {
    const uint32_t count = std::min<uint32_t>(
        model::ExecutionLimits::prefillTokenBudget,
        static_cast<uint32_t>(prompt.size()) - offset);
    BatchPlan plan{WorkKind::Prefill, BatchCohort::Greedy,
                   {{lane.id, count, offset}}, DecodeStage::Regular};
    ModelBatchItem item{lane.id, lane.slot, offset, offset, count,
                           lane.pages};
    item.inputTokens = prompt.subspan(offset, count);
    auto results =
        executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    if (results.size() != 1 || results[0].consumedPromptTokens != count)
      throw std::runtime_error("prefill consumed the wrong row count");
    offset += count;
  }
  lane.position = prompt.size();
}

struct CycleTiming final {
  double gpuSeconds = 0.0;
  double wallSeconds = 0.0;
  uint64_t commands = 0;
};

CycleTiming decodeCycle(metal::MetalBackend &backend,
                        model::Runtime &executor,
                        std::span<Lane> lanes) {
  const uint64_t submissionsBefore = backend.submissionCount();
  const auto started = std::chrono::steady_clock::now();
  BatchPlan plan;
  plan.kind = WorkKind::Decode;
  plan.cohort = BatchCohort::Greedy;
  std::vector<ModelBatchItem> items;
  for (Lane &lane : lanes) {
    plan.items.push_back({lane.id, 0, 0});
    items.push_back({lane.id, lane.slot, lane.position, 0, 0, lane.pages});
  }
  auto results = executor.decode(plan, items);
  if (results.size() != lanes.size())
    throw std::runtime_error("decode width changed");
  for (size_t index = 0; index < lanes.size(); ++index) {
    if (results[index].finished)
      throw std::runtime_error("the answer ended before profiling finished");
    lanes[index].position += results[index].outputTokens.size() -
                             results[index].outputTokensWithoutKv;
  }
  const auto finished = std::chrono::steady_clock::now();
  return {executor.telemetry().lastDecodeGpuSeconds,
          std::chrono::duration<double>(finished - started).count(),
          backend.submissionCount() - submissionsBefore};
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) {
        std::cerr << "usage: decode-profile METALLIB MODEL_ROOT "
                     "[--prompt-tokens N] [--cycles K] [--kv-format int8|bf16]\n";
        return 2;
      }
      uint32_t promptTokens = 512;
      uint32_t cycles = 4;
      kv::Format format = kv::Format::Int8;
      for (int index = 3; index < argc; index += 2) {
        const std::string_view option(argv[index]);
        if (index + 1 >= argc)
          throw std::invalid_argument(std::string(option) + " requires a value");
        if (option == "--prompt-tokens")
          promptTokens = parseCount(argv[index + 1], "--prompt-tokens");
        else if (option == "--cycles")
          cycles = parseCount(argv[index + 1], "--cycles");
        else if (option == "--kv-format") {
          const std::string_view value(argv[index + 1]);
          if (value != "int8" && value != "bf16")
            throw std::invalid_argument("--kv-format takes int8 or bf16");
          format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
        } else
          throw std::invalid_argument("unknown option");
      }

      metal::MetalBackend backend(argv[1]);
      model::ModelPackage model = model::loadModelPackage(
          backend, std::filesystem::path(argv[2]));
      ops::ExecutionPlans operators(backend.capabilities());
      model::ModelMemoryPlan executorPlan =
          model::plannedRuntimeMemory(backend.capabilities(), model, operators, format);

      // Enough Page32 pages for four lanes of prompt plus generated rows.
      const uint32_t pagesPerLane =
          (promptTokens + 256 + model::ExecutionLimits::targetVerifyRows) /
              kv::kPageTokens +
          2;
      const uint32_t pageCount =
          (pagesPerLane * 4 + model.targetKvLayout(format).sparseMappingBatchPages() -
           1) /
          model.targetKvLayout(format).sparseMappingBatchPages() *
          model.targetKvLayout(format).sparseMappingBatchPages();
      MemoryGovernor governor(
          backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
      kv::PageStorage pages(backend, governor.allocationAdmission(),
                              model.targetKvLayout(format), pageCount);
      for (uint32_t page = 0; page < pageCount; ++page) {
        if (!pages.ensureResident(page))
          throw std::runtime_error("could not back the KV pages");
      }
      model::QwenStateStorage states(backend,
                                      governor.allocationAdmission(),
                                      model.stateLayout());
      model::RuntimeContext context{
          backend, governor.allocationAdmission(), model, pages, states, operators,
          16384, executorPlan.pipelineReserveBytes,
          executorPlan.runtimeOverheadReserveBytes};
      model::Runtime executor(context);

      std::printf("device %s, %u prompt tokens, %u cycles per width\n",
                  backend.capabilities().deviceName.c_str(), promptTokens,
                  cycles);

      std::array<Lane, 4> lanes;
      for (uint32_t index = 0; index < lanes.size(); ++index) {
        lanes[index] = {index + 1, index, 0,
                        pageRange(index * pagesPerLane, pagesPerLane)};
      }
      const std::vector<uint32_t> prompt = chatPrompt(promptTokens);

      // Warm prefill and B1 decode with real work before measuring.
      prefill(executor, lanes[0], prompt);
      static_cast<void>(decodeCycle(backend, executor, std::span<Lane>(&lanes[0], 1)));
      executor.end(lanes[0].id);

      // Prefill: fused timing first, then the attributed replay. Both cover
      // every chunk of the prompt.
      const double prefillBefore = executor.telemetry().totalPrefillGpuSeconds;
      prefill(executor, lanes[0], prompt);
      const double prefillFused =
          executor.telemetry().totalPrefillGpuSeconds - prefillBefore;
      executor.end(lanes[0].id);
      backend.setDispatchProfiling(true);
      prefill(executor, lanes[0], prompt);
      backend.setDispatchProfiling(false);
      Table prefillTable;
      accumulate(prefillTable, backend.takeDispatchProfile());
      print("prefill " + std::to_string(promptTokens) + " rows", prefillTable,
            1.0, prefillFused);

      auto profileWidth = [&](const char *title, std::span<Lane> active) {
        std::vector<CycleTiming> fused;
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          fused.push_back(decodeCycle(backend, executor, active));
        std::sort(fused.begin(), fused.end(),
                  [](const CycleTiming &left, const CycleTiming &right) {
                    return left.gpuSeconds < right.gpuSeconds;
                  });
        const CycleTiming median = fused[fused.size() / 2];
        std::printf("\n%s: median fused gpu %.2f ms, wall %.2f ms, %llu "
                    "command(s) per cycle\n",
                    title, median.gpuSeconds * 1e3, median.wallSeconds * 1e3,
                    static_cast<unsigned long long>(median.commands));
        backend.setDispatchProfiling(true);
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          static_cast<void>(decodeCycle(backend, executor, active));
        backend.setDispatchProfiling(false);
        Table table;
        accumulate(table, backend.takeDispatchProfile());
        print(title, table, cycles, median.gpuSeconds);
      };
      profileWidth("B1 decode cycle", std::span<Lane>(&lanes[0], 1));
      // Add one lane at a time so M16 and M24 paths are measured too.
      for (uint32_t index = 1; index < lanes.size(); ++index) {
        prefill(executor, lanes[index], prompt);
        const std::string title = "B" + std::to_string(index + 1) + " decode cycle";
        profileWidth(title.c_str(), std::span<Lane>(lanes.data(), index + 1));
      }

      for (Lane &lane : lanes)
        executor.end(lane.id);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "decode-profile: " << error.what() << '\n';
      return 1;
    }
  }
}
