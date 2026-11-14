// Per-kernel GPU time attribution for the production executor.
//
//   decode-profile METALLIB MODEL_ROOT [--prompt-tokens N] [--cycles K] [--kv-format int8|bf16]
//                  [--state-format fp32|bf16] [--expert-memory SIZE] [--ple-memory SIZE]
//                  [--dispatch-list FILE] [--verify-rows auto|all]
//   decode-profile METALLIB MODEL_ROOT --images HxW[,HxW...] [--cycles K]
//
// Drives the real model runtime with Metal dispatch profiling enabled, so
// every dispatch of a prefill command and B1 through B4 DFlash cycles
// is replayed as its own command and attributed to its pipeline.
// The fused (unprofiled) GPU time of the same work is reported alongside, so
// separate replay changes scheduling and cache behavior. Its difference from
// normal command timing is not an exact decomposition of dispatch overhead. --dispatch-list
// writes the attributed B1 cycles' dispatches in order, one line each: index,
// pipeline and GPU microseconds.
// With --images it profiles the vision encoder instead, loading only the
// model's vision role: K encodes of a synthetic image per patch grid (height
// x width in patches), fused and then attributed.

#include "engine/Types.hpp"
#include "model/Runtime.hpp"
#include "ops/PageStorage.hpp"
#include "ops/Vision.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/SsdReaders.hpp"
#include "model/VisionLoader.hpp"
#include "model/WeightImages.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"
#include "tools/LoadOptions.hpp"
#include "tools/DecodeProfileOptions.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
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
// repeats one sentence, "Write a detailed, multi-paragraph essay about the
// history of the Roman Empire. ", until the prompt reaches the requested
// length, and the answer follows an empty think block. Its tokens are the
// target's tokenizer's: the Qwen families' (one tokenizer) or Flint's.
struct ChatTokens final {
  std::vector<uint32_t> userHeader;
  std::vector<uint32_t> sentence;
  std::vector<uint32_t> assistantHeader;
};
const ChatTokens kQwenChat{
    // <|im_start|>user\n
    .userHeader = {248045, 846, 198},
    .sentence = {7734, 264, 11346, 11, 7072, 12, 26829, 8627, 883, 279, 3712, 314, 279, 12386, 19825, 13, 220},
    // <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    .assistantHeader = {248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271}};
const ChatTokens kFlintChat{
    // <|user|>
    .userHeader = {150003},
    .sentence = {7421, 260, 7426, 11, 4519, 12, 41088, 6369, 762, 269, 3476, 297, 269, 10831, 18381, 13, 220},
    // <|assistant|><think></think>
    .assistantHeader = {150004, 150006, 150007}};

std::vector<uint32_t> chatPrompt(const model::ModelDescriptor &descriptor, uint32_t tokens) {
  const ChatTokens &chat =
      std::holds_alternative<model::FlintLayout>(descriptor.target) ? kFlintChat : kQwenChat;
  if (tokens < chat.userHeader.size() + chat.assistantHeader.size() + 1)
    throw std::invalid_argument("prompt is too short for a chat request");
  std::vector<uint32_t> prompt = chat.userHeader;
  const uint32_t body = tokens - chat.userHeader.size() - chat.assistantHeader.size();
  for (uint32_t index = 0; index < body; ++index)
    prompt.push_back(chat.sentence[index % chat.sentence.size()]);
  prompt.insert(prompt.end(), chat.assistantHeader.begin(), chat.assistantHeader.end());
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
             std::span<const uint32_t> prompt, uint32_t generationReserve) {
  EngineRequest request;
  request.id = lane.id;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = generationReserve;
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
  uint64_t tokens = 0;
};

CycleTiming decodeCycle(metal::MetalBackend &backend,
                        model::Runtime &executor,
                        std::span<Lane> lanes) {
  const uint64_t submissionsBefore =
      BackendInstrumentation::submittedCommands(backend);
  const auto started = std::chrono::steady_clock::now();
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
  uint64_t tokens = 0;
  for (size_t index = 0; index < lanes.size(); ++index) {
    tokens += results[index].outputTokens.size();
    if (!results[index].failure.empty())
      throw std::runtime_error(results[index].failure);
    if (results[index].finished)
      throw std::runtime_error("the answer ended before profiling finished");
    lanes[index].position += results[index].outputTokens.size() -
                             results[index].outputTokensWithoutKv;
  }
  const auto finished = std::chrono::steady_clock::now();
  return {executor.telemetry().lastDecodeGpuSeconds,
          std::chrono::duration<double>(finished - started).count(),
          BackendInstrumentation::submittedCommands(backend) - submissionsBefore, tokens};
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) {
        std::cerr << "usage: decode-profile METALLIB MODEL_ROOT "
                     "[--prompt-tokens N] [--cycles K] [--kv-format int8|bf16] "
                     "[--state-format fp32|bf16] [--expert-memory SIZE] [--ple-memory SIZE] "
                     "[--dispatch-list FILE] [--verify-rows auto|all] [--activations auto|bf16]\n"
                     "[--max-memory SIZE] [--max-context N] [--input-tokens FILE] [--report FILE]\n"
                     "[--require-flash-next] [--config-only]\n"
                     "       decode-profile METALLIB MODEL_ROOT "
                     "--images HxW[,HxW...] [--cycles K]\n";
        return 2;
      }
      std::vector<std::string_view> arguments;
      for (int i = 3; i < argc; ++i) arguments.emplace_back(argv[i]);
      const auto options = tools::DecodeProfileOptions::parse(arguments);
      const uint32_t promptTokens = options.promptTokens, cycles = options.cycles;
      const kv::Format format = options.kv == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
      const ops::GdnStateFormat stateFormat = options.state == "bf16" ? ops::GdnStateFormat::BFloat16 : ops::GdnStateFormat::Float32;
      const auto grids = options.images.empty() ? std::vector<ops::ImageGrid>{} : parseGrids(options.images);
      auto loadOptions = options.load;
      const std::string dispatchList = options.dispatchList;
      const model::VerifyRows verifyRows = options.verify == "all" ? model::VerifyRows::All : model::VerifyRows::Auto;
      std::cout << options.json() << '\n';
      if (options.configOnly) return 0; // Deliberately before Metal, model inspection or any output file.
      // All later observations are native fixture measurements, never HTTP TTFT.
      const std::filesystem::path root(argv[2]);
      std::optional<model::ModelDescriptor> checkedDescriptor;
      if (grids.empty()) {
        checkedDescriptor = model::inspectModelRoot(root);
        if (options.contextExplicit) *checkedDescriptor = model::servedFor(std::move(*checkedDescriptor), options.maxContext);
        if (options.requireFlashNext && !std::holds_alternative<model::Qwen4ExpLayout>(checkedDescriptor->target))
          throw std::invalid_argument("matched profile requires Flash-Next target geometry");
        if (uint64_t{promptTokens} + options.generationReserve() > checkedDescriptor->capabilities.maximumContextTokens)
          throw std::invalid_argument("profile exceeds the model's served context");
        const auto vocabulary = checkedDescriptor->capabilities.vocabularySize;
        if (std::any_of(options.tokens.begin(), options.tokens.end(), [vocabulary](auto t) { return t >= vocabulary; }))
          throw std::invalid_argument("token fixture is outside the model vocabulary");
        if (options.maxMemory && tools::weightBytes(root, *checkedDescriptor, loadOptions) > options.maxMemory)
          throw std::invalid_argument("model resident-weight estimate exceeds profiler memory limit");
      }
      metal::MetalBackend backend(argv[1]);
      if (!grids.empty()) {
        std::printf("device %s, %u encodes per image\n", backend.capabilities().deviceName.c_str(), cycles);
        profileImages(backend, root, grids, cycles);
        return 0;
      }
      loadOptions.readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
      auto descriptor = std::move(*checkedDescriptor);
      const uint64_t memoryLimit = options.maxMemory ? options.maxMemory : backend.capabilities().recommendedMaxWorkingSetBytes;
      if (tools::weightBytes(root, descriptor, loadOptions) > memoryLimit)
        throw std::invalid_argument("model resident-weight estimate exceeds profiler memory limit");
      model::LoadedModel model = model::loadModel(backend, root, descriptor, loadOptions);
      ops::ExecutionPlans operators(backend.capabilities(), options.activations == "bf16" ? ops::Int8Activations::Off : ops::Int8Activations::On);

      // Enough Page32 pages for every normal and attribution cycle at four widths.
      const uint32_t pagesPerLane =
          (promptTokens + options.generationReserve() + model::ExecutionLimits::targetVerifyRows) /
              kv::kPageTokens +
          2;
      // Whole extents of the largest size the pool rule picks, as a large
      // pool's would be.
      const kv::Layout kvLayout = model.targetKvLayout(format);
      const uint32_t extentPages = kvLayout.maximumExtentPages();
      const uint32_t pageCount =
          (pagesPerLane * 4 + extentPages - 1) / extentPages * extentPages;
      MemoryGovernor governor(
          backend, memoryLimit, 1, 1,
          queryHostMemory, 0);
      // Every allocation through the governor is for a claim: the profile's
      // are a start's, one Floor claim for its whole run.
      const MemoryGovernor::Scope run(governor, governor.ledger().open({.cls = ClaimClass::Floor}));
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
      context.verifyRows = verifyRows;
      model::Runtime executor(context);

      std::printf("device %s, %u prompt tokens, %u cycles per width, %s state\n",
                  backend.capabilities().deviceName.c_str(), promptTokens,
                  cycles, std::string(ops::formatName(stateFormat)).c_str());

      std::array<Lane, 4> lanes;
      for (uint32_t index = 0; index < lanes.size(); ++index) {
        lanes[index] = {index + 1, index, 0,
                        pageRange(index * pagesPerLane, pagesPerLane)};
      }
      const std::vector<uint32_t> prompt = options.tokens.empty() ? chatPrompt(model.descriptor, promptTokens) : options.tokens;

      // Warm prefill and B1 decode with real work before measuring.
      prefill(executor, lanes[0], prompt, options.generationReserve());
      static_cast<void>(decodeCycle(backend, executor, std::span<Lane>(&lanes[0], 1)));
      executor.end(lanes[0].id);

      // Prefill: fused timing first, then the attributed replay. Both cover
      // every chunk of the prompt.
      const double prefillBefore = executor.telemetry().totalPrefillGpuSeconds;
      prefill(executor, lanes[0], prompt, options.generationReserve());
      const double prefillFused =
          executor.telemetry().totalPrefillGpuSeconds - prefillBefore;
      executor.end(lanes[0].id);
      BackendInstrumentation::setDispatchProfiling(backend, true);
      prefill(executor, lanes[0], prompt, options.generationReserve());
      BackendInstrumentation::setDispatchProfiling(backend, false);
      Table prefillTable;
      accumulate(prefillTable,
                 BackendInstrumentation::takeDispatchProfile(backend));
      print("prefill " + std::to_string(promptTokens) + " rows", prefillTable,
            1.0, prefillFused);

      std::ostringstream cycleReport;
      cycleReport << "[";
      bool firstWidth = true;
      auto profileWidth = [&](const char *title, std::span<Lane> active) {
        std::vector<CycleTiming> fused;
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          fused.push_back(decodeCycle(backend, executor, active));
        std::sort(fused.begin(), fused.end(),
                  [](const CycleTiming &left, const CycleTiming &right) {
                    return left.gpuSeconds < right.gpuSeconds;
                  });
        const CycleTiming median = fused[fused.size() / 2];
        double tokens = 0.0, wall = 0.0;
        for (const CycleTiming &cycle : fused) {
          tokens += cycle.tokens;
          wall += cycle.wallSeconds;
        }
        std::printf("\n%s: median fused gpu %.2f ms, wall %.2f ms, %llu "
                    "command(s) per cycle, %.2f tokens per cycle, %.2f wall ms per token\n",
                    title, median.gpuSeconds * 1e3, median.wallSeconds * 1e3,
                    static_cast<unsigned long long>(median.commands), tokens / fused.size(),
                    wall * 1e3 / tokens);
        if (!std::isfinite(wall) || !std::isfinite(median.gpuSeconds) || wall < 0 || median.gpuSeconds < 0)
          throw std::runtime_error("invalid native timing observation");
        if (!firstWidth) cycleReport << ',';
        firstWidth = false;
        cycleReport << std::setprecision(17) << "{\"lanes\":" << active.size() << ",\"gpu_cycle_median_s\":" << median.gpuSeconds
                    << ",\"cycle_wall_s\":" << wall << ",\"emitted_tokens\":" << tokens
                    << ",\"output_tps\":" << (wall > 0 ? tools::profileNumber(tokens / wall) : "null")
                    << ",\"commands_per_median_cycle\":" << median.commands << ",\"cpu_graph_s\":null}";
        BackendInstrumentation::setDispatchProfiling(backend, true);
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          static_cast<void>(decodeCycle(backend, executor, active));
        BackendInstrumentation::setDispatchProfiling(backend, false);
        const std::vector<metal::DispatchTiming> timings =
            BackendInstrumentation::takeDispatchProfile(backend);
        if (!dispatchList.empty() && active.size() == 1) {
          std::ostringstream trace;
          trace << std::fixed << std::setprecision(2);
          for (size_t index = 0; index < timings.size(); ++index)
            trace << index << ' ' << timings[index].pipelineName << ' '
                  << timings[index].gpuSeconds * 1e6 << '\n';
          tools::writePrivateProfileOutput(dispatchList, trace.str());
        }
        Table table;
        accumulate(table, timings);
        print(title, table, cycles, median.gpuSeconds);
      };
      profileWidth("B1 decode cycle", std::span<Lane>(&lanes[0], 1));
      // Add one lane at a time so M16 and M24 paths are measured too.
      for (uint32_t index = 1; index < lanes.size(); ++index) {
        prefill(executor, lanes[index], prompt, options.generationReserve());
        const std::string title = "B" + std::to_string(index + 1) + " decode cycle";
        profileWidth(title.c_str(), std::span<Lane>(lanes.data(), index + 1));
      }

      for (Lane &lane : lanes)
        executor.end(lane.id);
      if (!options.report.empty()) {
        const std::string json = "{\"configuration\":" + options.json() + ",\"model_source_identity\":" +
            tools::profileJsonString(model.descriptor.sourceIdentity) + ",\"effective_memory_limit_bytes\":" + std::to_string(memoryLimit) +
            ",\"effective_max_context_tokens\":" + std::to_string(model.descriptor.capabilities.maximumContextTokens) + ",\"prefill_fused_gpu_s\":" + tools::profileNumber(prefillFused) + ",\"native_cycles\":" + cycleReport.str() + "]}";
        tools::writePrivateProfileOutput(options.report, json + "\n");
      }
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "decode-profile: " << error.what() << '\n';
      return 1;
    }
  }
}
