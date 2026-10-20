// Per-kernel GPU time attribution for the production executor.
//
//   decode-profile METALLIB MODEL_ROOT [--prompt-tokens N] [--cycles K] [--kv-format int8|bf16]
//                  [--state-format fp32|bf16] [--live-verify-threshold Q]
//   decode-profile METALLIB MODEL_ROOT --images HxW[,HxW...] [--cycles K]
//
// Drives the real model runtime with Metal dispatch profiling enabled, so
// every dispatch of a prefill command and B1 through B4 DFlash cycles
// is replayed as its own command and attributed to its pipeline.
// The fused (unprofiled) GPU time of the same work is reported alongside, so
// the gap between the sum of parts and the fused command shows how much a
// cycle pays in dispatch boundaries rather than kernel work.
// With --images it profiles the vision encoder instead, loading only the
// model's vision role: K encodes of a synthetic image per patch grid (height
// x width in patches), fused and then attributed.

#include "engine/Types.hpp"
#include "AwakeClock.hpp"
#include "model/Runtime.hpp"
#include "ops/PageStorage.hpp"
#include "ops/Vision.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/VisionLoader.hpp"
#include "model/WeightImages.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::engine;
using metal::BackendInstrumentation;

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

// HxW[,HxW...]: patch grids of images the encoder accepts.
std::vector<ops::ImageGrid> parseGrids(std::string_view text) {
  std::vector<ops::ImageGrid> grids;
  while (true) {
    const std::string_view item = text.substr(0, text.find(','));
    const size_t cross = item.find('x');
    if (cross == std::string_view::npos)
      throw std::invalid_argument("--images takes HxW[,HxW...] patch grids");
    const ops::ImageGrid grid{parseCount(item.substr(0, cross), "grid height"),
                              parseCount(item.substr(cross + 1), "grid width")};
    if (!grid.valid() || grid.patches() > ops::kMaximumImagePatches) {
      throw std::invalid_argument(
          std::string(item) + " is not a grid of even sides and at most " +
          std::to_string(ops::kMaximumImagePatches) + " patches");
    }
    grids.push_back(grid);
    if (item.size() == text.size())
      return grids;
    text.remove_prefix(item.size() + 1);
  }
}

// The vision encoder as the runtime runs it for the largest of the images:
// an arena sized for that image's patches, and each image's dispatches in
// one command. Every image is encoded once to warm up, then `cycles` times
// fused for the median GPU time and `cycles` times attributed.
void profileImages(metal::MetalBackend &backend,
                   const std::filesystem::path &root,
                   std::span<const ops::ImageGrid> grids, uint32_t cycles) {
  const model::ModelDescriptor descriptor = model::inspectModelRoot(root);
  if (!descriptor.hasVision())
    throw std::invalid_argument("the model has no vision role");
  model::WeightImages images(backend, descriptor.sourceIdentity);
  const auto loader = model::planVisionLoader(root, descriptor);
  const model::QwenVisionWeights vision =
      model::loadVisionWeights(backend, images, loader.get());
  uint64_t largest = 0;
  for (const ops::ImageGrid &grid : grids)
    largest = std::max(largest, grid.patches());
  const ops::Vision encoder(backend, vision.tensors,
                            static_cast<uint32_t>(largest));

  for (const ops::ImageGrid &grid : grids) {
    // Gradients along both axes and a checkerboard of patch-sized squares.
    const metal::MetalBuffer pixels = backend.allocateBuffer(
        grid.pixelBytes(), metal::BufferStorage::Shared, "image pixels");
    const uint32_t height = grid.height * ops::kImagePatchPixels;
    const uint32_t width = grid.width * ops::kImagePatchPixels;
    auto *pixel = static_cast<uint8_t *>(pixels.contents());
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x, pixel += ops::kImageChannels) {
        pixel[0] = static_cast<uint8_t>(x * 255 / width);
        pixel[1] = static_cast<uint8_t>(y * 255 / height);
        pixel[2] = static_cast<uint8_t>(
            (x / ops::kImagePatchPixels + y / ops::kImagePatchPixels) % 2 * 255);
      }
    }
    const metal::MetalBuffer embeddings = backend.allocateBuffer(
        uint64_t{ops::Vision::embeddingRows(grid)} *
            vision.tensors.layout.outputHiddenSize * sizeof(uint16_t),
        metal::BufferStorage::Private, "image embeddings");
    const auto encode = [&] {
      metal::CommandGraph graph;
      encoder.encode(graph, grid, pixels, embeddings);
      return backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
    };

    static_cast<void>(encode());
    std::vector<double> fused;
    for (uint32_t cycle = 0; cycle < cycles; ++cycle)
      fused.push_back(encode());
    std::sort(fused.begin(), fused.end());
    BackendInstrumentation::setDispatchProfiling(backend, true);
    for (uint32_t cycle = 0; cycle < cycles; ++cycle)
      static_cast<void>(encode());
    BackendInstrumentation::setDispatchProfiling(backend, false);
    Table table;
    accumulate(table, BackendInstrumentation::takeDispatchProfile(backend));
    print("image " + std::to_string(grid.height) + "x" +
              std::to_string(grid.width) + " (" +
              std::to_string(grid.patches()) + " patches) encode",
          table, cycles, fused[fused.size() / 2]);
  }
}

struct Lane final {
  uint64_t id = 0;
  uint32_t stateLane = 0;
  uint64_t position = 0;
  std::vector<uint32_t> pages;
  // The pages never change, so the page table keeps its first revision.
  uint64_t pageTableRevision = 1;
};

void prefill(model::Runtime &executor, Lane &lane,
             std::span<const uint32_t> prompt, uint32_t generatedBudget) {
  EngineRequest request;
  request.id = lane.id;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = generatedBudget;
  executor.beginColdRequest(request.modelView(), lane.stateLane);
  uint32_t offset = 0;
  while (offset < prompt.size()) {
    const uint32_t count = std::min<uint32_t>(
        model::ExecutionLimits::prefillTokenBudget,
        static_cast<uint32_t>(prompt.size()) - offset);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{lane.id, count, offset}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{lane.id, offset, count, lane.pages,
                        lane.pageTableRevision};
    item.inputTokens = prompt.subspan(offset, count);
    auto results =
        executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    if (results.size() != 1 || results[0].consumedPromptTokens != count)
      throw std::runtime_error("prefill consumed the wrong row count");
    if (!results[0].failure.empty())
      throw std::runtime_error(results[0].failure);
    offset += count;
  }
  lane.position = prompt.size();
}

struct CycleTiming final {
  double gpuSeconds = 0.0;
  double wallSeconds = 0.0;
  uint64_t commands = 0;
  uint32_t liveRows = 0;
  uint32_t tokens = 0;
  uint32_t proposed = 0;
  uint32_t accepted = 0;
  std::array<uint32_t, model::ExecutionLimits::draftProposalTokens> survival{};
  double start = 0.0;
  double end = 0.0;
  std::vector<metal::CommandTrace> trace;
};

CycleTiming decodeCycle(metal::MetalBackend &backend,
                        model::Runtime &executor,
                        std::span<Lane> lanes) {
  const uint64_t submissionsBefore =
      BackendInstrumentation::submittedCommands(backend);
  const auto started = AwakeClock::now();
  BatchPlan plan;
  plan.kind = WorkKind::Decode;
  std::vector<ModelBatchItem> items;
  for (Lane &lane : lanes) {
    plan.items.push_back({lane.id, 0, 0});
    items.push_back({lane.id, lane.position, 0, lane.pages,
                     lane.pageTableRevision});
  }
  auto results = executor.decode(plan, items);
  if (results.size() != lanes.size())
    throw std::runtime_error("decode width changed");
  uint32_t tokens = 0;
  uint32_t proposed = 0, accepted = 0;
  std::array<uint32_t, model::ExecutionLimits::draftProposalTokens> survival{};
  for (size_t index = 0; index < lanes.size(); ++index) {
    if (!results[index].failure.empty())
      throw std::runtime_error(results[index].failure);
    if (results[index].finished)
      throw std::runtime_error("the answer ended before profiling finished");
    tokens += static_cast<uint32_t>(results[index].outputTokens.size());
    proposed += results[index].draftedTokens;
    accepted += results[index].acceptedDraftTokens;
    for (uint32_t position = 0; position < results[index].acceptedDraftTokens; ++position)
      ++survival.at(position);
    lanes[index].position += results[index].outputTokens.size() -
                             results[index].outputTokensWithoutKv;
  }
  const auto finished = AwakeClock::now();
  return {executor.telemetry().lastDecodeGpuSeconds,
          std::chrono::duration<double>(finished - started).count(),
          BackendInstrumentation::submittedCommands(backend) - submissionsBefore,
          executor.telemetry().lastDecodeLiveRows, tokens, proposed, accepted, survival,
          std::chrono::duration<double>(started.time_since_epoch()).count(),
          std::chrono::duration<double>(finished.time_since_epoch()).count(),
          BackendInstrumentation::takeCommandTrace(backend)};
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      std::setvbuf(stdout, nullptr, _IOLBF, 0);
      if (argc < 3) {
        std::cerr << "usage: decode-profile METALLIB MODEL_ROOT "
                     "[--prompt-tokens N] [--cycles K] [--kv-format int8|bf16] "
                     "[--state-format fp32|bf16] [--live-verify-threshold Q] [--live-gdn on|off] "
                     "[--attribution on|off] [--json-output PATH] [--trace-output PATH]\n"
                     "       decode-profile METALLIB MODEL_ROOT "
                     "--images HxW[,HxW...] [--cycles K]\n";
        return 2;
      }
      uint32_t promptTokens = 512;
      uint32_t cycles = 4;
      kv::Format format = kv::Format::Int8;
      ops::GdnStateFormat stateFormat = ops::GdnStateFormat::Float32;
      std::vector<ops::ImageGrid> grids;
      bool textOptions = false;
      float liveVerifyThreshold = 0.0F;
      bool liveGdn = false;
      bool attribution = true;
      std::string jsonPath, tracePath;
      for (int index = 3; index < argc; index += 2) {
        const std::string_view option(argv[index]);
        if (index + 1 >= argc)
          throw std::invalid_argument(std::string(option) + " requires a value");
        if (option == "--prompt-tokens") {
          promptTokens = parseCount(argv[index + 1], "--prompt-tokens");
          textOptions = true;
        } else if (option == "--cycles")
          cycles = parseCount(argv[index + 1], "--cycles");
        else if (option == "--live-gdn") {
          const std::string_view value(argv[index + 1]);
          if (value != "on" && value != "off")
            throw std::invalid_argument("--live-gdn takes on or off");
          liveGdn = value == "on";
        } else if (option == "--live-verify-threshold") {
          textOptions = true;
          size_t consumed = 0;
          const std::string value(argv[index + 1]);
          liveVerifyThreshold = std::stof(value, &consumed);
          if (consumed != value.size() || !std::isfinite(liveVerifyThreshold) ||
              liveVerifyThreshold < 0.0F || liveVerifyThreshold > 1.0F)
            throw std::invalid_argument("invalid live verify threshold");
        } else if (option == "--attribution") {
          const std::string_view value(argv[index + 1]);
          if (value != "on" && value != "off")
            throw std::invalid_argument("--attribution takes on or off");
          attribution = value == "on";
          textOptions = true;
        } else if (option == "--json-output" || option == "--trace-output") {
          (option == "--json-output" ? jsonPath : tracePath) = argv[index + 1];
          textOptions = true;
        } else if (option == "--kv-format") {
          const std::string_view value(argv[index + 1]);
          if (value != "int8" && value != "bf16")
            throw std::invalid_argument("--kv-format takes int8 or bf16");
          format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
          textOptions = true;
        } else if (option == "--state-format") {
          const std::string_view value(argv[index + 1]);
          if (value != "fp32" && value != "bf16")
            throw std::invalid_argument("--state-format takes fp32 or bf16");
          stateFormat = value == "fp32" ? ops::GdnStateFormat::Float32 : ops::GdnStateFormat::BFloat16;
          textOptions = true;
        } else if (option == "--images")
          grids = parseGrids(argv[index + 1]);
        else
          throw std::invalid_argument("unknown option");
      }
      if (!grids.empty() && textOptions)
        throw std::invalid_argument("--images takes no other option than --cycles");

      const uint64_t generatedBound = std::max<uint64_t>(256,
          uint64_t{4} * (uint64_t{cycles} * 2 + 2) * model::ExecutionLimits::targetVerifyRows + 1);
      if (uint64_t{promptTokens} + generatedBound + model::ExecutionLimits::targetVerifyRows >
          kv::kMaximumPhysicalTokens)
        throw std::invalid_argument("profile prompt and cycle budget exceed physical context");
      const uint32_t generatedBudget = static_cast<uint32_t>(generatedBound);
      std::ofstream json, trace;
      if (!jsonPath.empty()) {
        json.open(jsonPath);
        if (!json) throw std::runtime_error("cannot open JSON output");
        json << std::setprecision(17) << "[\n";
      }
      if (!tracePath.empty()) {
        trace.open(tracePath);
        if (!trace) throw std::runtime_error("cannot open trace output");
        trace << std::setprecision(17) << "{\"traceEvents\":[\n";
      }
      bool firstSample = true, firstEvent = true;
      const double traceOrigin = std::chrono::duration<double>(AwakeClock::now().time_since_epoch()).count();
      auto event = [&](const char *name, uint32_t tid, double begin, double end) {
        if (!trace.is_open() || !std::isfinite(begin) || !std::isfinite(end) || end < begin) return;
        trace << (firstEvent ? "" : ",\n") << "{\"name\":\"" << name
              << "\",\"ph\":\"X\",\"pid\":1,\"tid\":" << tid
              << ",\"ts\":" << (begin - traceOrigin) * 1e6 << ",\"dur\":" << (end - begin) * 1e6 << '}';
        firstEvent = false;
      };

      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path root(argv[2]);
      if (!grids.empty()) {
        std::printf("device %s, %u encodes per image\n",
                    backend.capabilities().deviceName.c_str(), cycles);
        profileImages(backend, root, grids, cycles);
        return 0;
      }
      model::LoadedModel model =
          model::loadModel(backend, root, model::inspectModelRoot(root));
      ops::ExecutionPlans operators(backend.capabilities());

      // Enough Page32 pages for four lanes of prompt plus generated rows.
      const uint32_t pagesPerLane =
          (promptTokens + generatedBudget + model::ExecutionLimits::targetVerifyRows) /
              kv::kPageTokens +
          2;
      // Whole extents of the largest size the pool rule picks, as a large
      // pool's would be.
      const kv::Layout kvLayout = model.targetKvLayout(format);
      const uint32_t extentPages = kvLayout.maximumExtentPages();
      const uint32_t pageCount =
          (pagesPerLane * 4 + extentPages - 1) / extentPages * extentPages;
      MemoryGovernor governor(
          backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
          queryHostMemory, 0);
      kv::PageStorage pages(backend, governor.allocationAdmission(), kvLayout,
                            pageCount, extentPages);
      for (uint32_t extent = 0; extent < pageCount / extentPages; ++extent) {
        if (!pages.allocateExtent(extent))
          throw std::runtime_error("could not allocate the KV extents");
      }
      model::QwenStateStorage states(backend,
                                      governor.allocationAdmission(),
                                      model.stateLayout(stateFormat), nullptr);
      model::RuntimeContext context{backend, model, pages, states, operators};
      model::Runtime executor(context);
      executor.setLiveVerifyThreshold(liveVerifyThreshold);
      executor.setLiveGdn(liveGdn);
      std::printf("live verify threshold %.6g (greedy MoE only)\n", liveVerifyThreshold);

      std::printf("device %s, %u prompt tokens, %u cycles per width, %s state\n",
                  backend.capabilities().deviceName.c_str(), promptTokens,
                  cycles, std::string(ops::formatName(stateFormat)).c_str());

      std::array<Lane, 4> lanes;
      for (uint32_t index = 0; index < lanes.size(); ++index) {
        lanes[index] = {index + 1, index, 0,
                        pageRange(index * pagesPerLane, pagesPerLane)};
      }
      const std::vector<uint32_t> prompt = chatPrompt(promptTokens);

      // Warm prefill and B1 decode with real work before measuring.
      prefill(executor, lanes[0], prompt, generatedBudget);
      static_cast<void>(decodeCycle(backend, executor, std::span<Lane>(&lanes[0], 1)));
      executor.end(lanes[0].id);

      // Prefill: fused timing first, then the attributed replay. Both cover
      // every chunk of the prompt.
      const double prefillBefore = executor.telemetry().totalPrefillGpuSeconds;
      prefill(executor, lanes[0], prompt, generatedBudget);
      const double prefillFused =
          executor.telemetry().totalPrefillGpuSeconds - prefillBefore;
      executor.end(lanes[0].id);
      if (attribution) {
        BackendInstrumentation::setDispatchProfiling(backend, true);
        prefill(executor, lanes[0], prompt, generatedBudget);
        BackendInstrumentation::setDispatchProfiling(backend, false);
        Table prefillTable;
        accumulate(prefillTable,
                   BackendInstrumentation::takeDispatchProfile(backend));
        print("prefill " + std::to_string(promptTokens) + " rows", prefillTable,
              1.0, prefillFused);
      } else {
        prefill(executor, lanes[0], prompt, generatedBudget);
      }

      auto profileWidth = [&](const char *title, std::span<Lane> active) {
        // Compile each width's pipelines before measuring or tracing it.
        static_cast<void>(decodeCycle(backend, executor, active));
        BackendInstrumentation::setCommandTracing(backend, !tracePath.empty());
        std::vector<CycleTiming> fused;
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          fused.push_back(decodeCycle(backend, executor, active));
        BackendInstrumentation::setCommandTracing(backend, false);
        for (uint32_t sample = 0; sample < fused.size(); ++sample) {
          const CycleTiming &cycle = fused[sample];
          if (json.is_open()) {
            json << (firstSample ? "" : ",\n") << "{\"width\":" << active.size()
                 << ",\"sample\":" << sample << ",\"prompt_tokens\":" << promptTokens
                 << ",\"live_gdn\":" << (liveGdn ? "true" : "false")
                 << ",\"threshold\":" << liveVerifyThreshold
                 << ",\"gpu_seconds\":" << cycle.gpuSeconds
                 << ",\"wall_seconds\":" << cycle.wallSeconds
                 << ",\"commands\":" << cycle.commands << ",\"eligible_rows\":" << cycle.liveRows
                 << ",\"committed_tokens\":" << cycle.tokens << ",\"proposed_tokens\":" << cycle.proposed
                 << ",\"accepted_tokens\":" << cycle.accepted << ",\"prefix_survival\":[";
            for (uint32_t position = 0; position < cycle.survival.size(); ++position)
              json << (position ? "," : "") << cycle.survival[position];
            json << "]}";
            firstSample = false;
          }
          event(title, 0, cycle.start, cycle.end);
          double begin = cycle.start;
          for (const metal::CommandTrace &command : cycle.trace) {
            event("graph construction", 1, begin, command.prepareStart);
            event("backend preparation", 1, command.prepareStart, command.encodeStart);
            event("Metal encoding", 1, command.encodeStart, command.commit);
            event("GPU execution", 2, command.gpuStart, command.gpuEnd);
            event("completion notification", 1, command.gpuEnd, command.complete);
            begin = command.complete;
          }
          event("result finalization", 1, begin, cycle.end);
        }
        std::sort(fused.begin(), fused.end(),
                  [](const CycleTiming &left, const CycleTiming &right) {
                    return left.gpuSeconds < right.gpuSeconds;
                  });
        const CycleTiming median = fused[fused.size() / 2];
        std::printf("\n%s: median fused gpu %.2f ms, wall %.2f ms, %llu "
                    "command(s) per cycle\n",
                    title, median.gpuSeconds * 1e3, median.wallSeconds * 1e3,
                    static_cast<unsigned long long>(median.commands));
        uint64_t liveRows = 0, produced = 0;
        double wall = 0.0;
        for (const CycleTiming &cycle : fused) {
          liveRows += cycle.liveRows;
          produced += cycle.tokens;
          wall += cycle.wallSeconds;
        }
        std::printf("mean live rows %.2f/%zu, tokens/cycle %.2f, fused wall %.2f tok/s\n",
                    double(liveRows) / cycles, active.size() * model::ExecutionLimits::targetVerifyRows,
                    double(produced) / cycles, double(produced) / wall);
        if (attribution) {
          BackendInstrumentation::setDispatchProfiling(backend, true);
          for (uint32_t cycle = 0; cycle < cycles; ++cycle)
            static_cast<void>(decodeCycle(backend, executor, active));
          BackendInstrumentation::setDispatchProfiling(backend, false);
          Table table;
          accumulate(table,
                     BackendInstrumentation::takeDispatchProfile(backend));
          print(title, table, cycles, median.gpuSeconds);
        }
      };
      profileWidth("B1 decode cycle", std::span<Lane>(&lanes[0], 1));
      // Add one lane at a time so M16 and M24 paths are measured too.
      for (uint32_t index = 1; index < lanes.size(); ++index) {
        prefill(executor, lanes[index], prompt, generatedBudget);
        const std::string title = "B" + std::to_string(index + 1) + " decode cycle";
        profileWidth(title.c_str(), std::span<Lane>(lanes.data(), index + 1));
      }

      for (Lane &lane : lanes)
        executor.end(lane.id);
      if (json.is_open()) {
        json << "\n]\n";
        if (!json) throw std::runtime_error("cannot write JSON output");
      }
      if (trace.is_open()) {
        trace << "\n]}\n";
        if (!trace) throw std::runtime_error("cannot write trace output");
      }
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "decode-profile: " << error.what() << '\n';
      return 1;
    }
  }
}
