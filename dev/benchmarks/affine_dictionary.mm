// Native coefficient dictionary experiment. Serving loaders never attach dictionaries.
#include "TranscriptPrompts.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "model/Qwen3_8.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/PageStorage.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <variant>

using namespace splash;
using Clock = std::chrono::steady_clock;

namespace {
uint32_t count(std::string_view text) {
  uint32_t result = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("expected an unsigned integer");
  return result;
}

template <class T> void array(std::span<const T> values) {
  std::cout << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

struct Cycle {
  uint32_t rows, emitted, accepted;
  double wall, gpu;
};

struct Result {
  double wall = 0.0, ttft = 0.0;
  std::vector<uint32_t> output;
  std::vector<Cycle> cycles;
};

Result run(model::Runtime &runtime, std::span<const uint32_t> prompt, uint32_t outputTokens,
           const std::vector<uint32_t> &pages) {
  engine::EngineRequest request;
  request.id = 1;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = outputTokens;
  request.sampling.temperature = 0.0F;
  runtime.beginColdRequest(request.modelView(), 0);
  Result result;
  const auto started = Clock::now();
  uint32_t offset = 0;
  while (offset < prompt.size()) {
    const uint32_t rows = std::min<uint32_t>(model::ExecutionLimits::prefillTokenBudget, prompt.size() - offset);
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{1, rows, offset}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{1, offset, rows, pages, 1};
    item.inputTokens = prompt.subspan(offset, rows);
    const auto done = runtime.prefill(plan, std::span(&item, 1));
    if (done.size() != 1 || done[0].consumedPromptTokens != rows || !done[0].failure.empty())
      throw std::runtime_error("prefill failed");
    offset += rows;
  }
  uint64_t position = prompt.size();
  while (result.output.size() < outputTokens) {
    BatchPlan plan{.kind = WorkKind::Decode, .items = {{1, 0, 0}}};
    ModelBatchItem item{1, position, 0, pages, 1};
    const auto before = Clock::now();
    const auto done = runtime.decode(plan, std::span(&item, 1));
    const double wall = std::chrono::duration<double>(Clock::now() - before).count();
    if (done.size() != 1) throw std::runtime_error("decode width changed");
    if (!done[0].failure.empty()) throw std::runtime_error("decode failed: " + done[0].failure);
    const auto &step = done[0];
    if (result.output.empty()) result.ttft = std::chrono::duration<double>(Clock::now() - started).count();
    result.output.insert(result.output.end(), step.outputTokens.begin(), step.outputTokens.end());
    position += step.outputTokens.size() - step.outputTokensWithoutKv;
    if (step.outputTokens.size() > step.outputTokensWithoutKv)
      result.cycles.push_back({model::ExecutionLimits::targetVerifyRows,
                               uint32_t(step.outputTokens.size()), step.acceptedDraftTokens,
                               wall, runtime.telemetry().lastDecodeGpuSeconds});
    if (step.finished) break;
    if (step.outputTokens.empty()) throw std::runtime_error("decode made no progress");
  }
  result.wall = std::chrono::duration<double>(Clock::now() - started).count();
  runtime.end(1);
  return result;
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) throw std::invalid_argument(
          "usage: affine-dictionary-experiment METALLIB MODEL_ROOT [--dictionary 0|1] "
          "[--audit 0|1] [--prompt-index 0..5] [--tokens-file whitespace_u32_file] [--output-tokens N]");
      bool dictionary = false, audit = false;
      std::optional<uint32_t> promptIndex;
      std::optional<std::string> tokenFile;
      uint32_t outputTokens = 256;
      for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("option requires a value");
        const std::string_view option(argv[i]), value(argv[i + 1]);
        if (option == "--dictionary" || option == "--audit") {
          if (value != "0" && value != "1") throw std::invalid_argument("flag takes 0 or 1");
          (option == "--dictionary" ? dictionary : audit) = value == "1";
        } else if (option == "--prompt-index") promptIndex = count(value);
        else if (option == "--tokens-file") tokenFile = value;
        else if (option == "--output-tokens") outputTokens = count(value);
        else throw std::invalid_argument("unknown option");
      }
      if (!outputTokens || outputTokens > 4096 || (promptIndex && *promptIndex >= 6) ||
          (tokenFile && promptIndex) || (audit && !dictionary)) throw std::invalid_argument("invalid request options");
      std::vector<std::vector<uint32_t>> prompts;
      if (tokenFile) {
        std::ifstream input(*tokenFile);
        if (!input) throw std::invalid_argument("cannot open tokens file");
        prompts.emplace_back();
        std::string encoded;
        while (input >> encoded) prompts.back().push_back(count(encoded));
        if (!input.eof() || prompts.back().empty()) throw std::invalid_argument("invalid tokens file");
      } else {
        for (uint32_t index = 0; index < benchmark::kTranscriptPrompts.size(); ++index)
          if (!promptIndex || index == *promptIndex) {
            const auto prompt = benchmark::kTranscriptPrompts[index];
            prompts.emplace_back(prompt.begin(), prompt.end());
          }
      }
      const uint64_t maxPrompt = std::max_element(prompts.begin(), prompts.end(),
          [](const auto &a, const auto &b) { return a.size() < b.size(); })->size();
      if (maxPrompt + outputTokens + 8 > 32768) throw std::invalid_argument("experiment context exceeds 32768");
      const auto preparing = Clock::now();
      metal::MetalBackend backend(argv[1]);
      const auto descriptor = model::inspectModelRoot(argv[2]);
      const uint32_t vocabulary = std::visit([](const auto &layout) { return layout.vocabularySize; }, descriptor.target);
      for (const auto &prompt : prompts)
        if (std::any_of(prompt.begin(), prompt.end(), [&](uint32_t token) { return token >= vocabulary; }))
          throw std::invalid_argument("prompt token exceeds target vocabulary");
      auto loaded = model::loadModel(backend, argv[2], descriptor);
      auto *weights = std::get_if<model::Qwen3_8Weights>(&loaded.target);
      if (!weights) throw std::invalid_argument("experiment needs a Qwen3.8 dense target");
      uint32_t modules = 0;
      uint64_t units = 0, payload = 0, uniquePairs = 0;
      const auto converting = Clock::now();
      if (dictionary) {
        for (auto &layer : weights->layers)
          for (auto *projection : {&layer.gateProjection, &layer.upProjection, &layer.downProjection}) {
            *projection = projection->withAffineCoefficientDictionary(backend);
            if (projection->blocks().segments.size() != 1) continue;
            const auto &segment = projection->blocks().segments.front();
            const auto &d = segment.coefficientDictionary;
            if (!d) continue;
            ++modules;
            units += d->units;
            uniquePairs += d->entries;
            payload += d->indices.sizeBytes() + d->pairs.sizeBytes();
            if (audit) {
              auto restored = backend.allocateBuffer(d->units * 4, metal::BufferStorage::Shared,
                                                      "dictionary reconstruction audit");
              metal::CommandGraph graph;
              graph.add("experiment_affine_dictionary_roundtrip", {d->indices, d->pairs, restored}, uint32_t(d->units),
                        {uint32_t((d->units + 255) / 256), 1, 1}, {256, 1, 1});
              (void)backend.submitCommandAsync(graph.dispatches()).wait();
              if (std::memcmp(restored.contents(), segment.meta.contents(), d->units * 4))
                throw std::runtime_error("GPU reconstruction differs from original coefficient bits");
            }
          }
        if (!modules) throw std::invalid_argument("no supported AF4G64 MLP projections were converted");
      }
      const double conversion = std::chrono::duration<double>(Clock::now() - converting).count();
      ops::ExecutionPlans operators(backend.capabilities());
      const auto layout = loaded.targetKvLayout(kv::Format::Int8);
      const uint32_t extent = layout.maximumExtentPages();
      const uint32_t needed = (maxPrompt + outputTokens + 8 + kv::kPageTokens - 1) / kv::kPageTokens + 2;
      const uint32_t pageCount = (needed + extent - 1) / extent * extent;
      engine::MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                                      engine::queryHostMemory, 0);
      kv::PageStorage storage(backend, governor.allocationAdmission(), layout, pageCount, extent);
      for (uint32_t i = 0; i < pageCount / extent; ++i)
        if (!storage.allocateExtent(i)) throw std::runtime_error("KV allocation failed");
      model::QwenStateStorage states(backend, governor.allocationAdmission(),
                                    loaded.stateLayout(ops::GdnStateFormat::Float32), nullptr);
      model::RuntimeContext context{backend, loaded, storage, states, operators};
      context.verifyRows = model::VerifyRows::All;
      model::Runtime runtime(context);
      std::vector<uint32_t> pages(pageCount);
      std::iota(pages.begin(), pages.end(), 0);
      const double preparation = std::chrono::duration<double>(Clock::now() - preparing).count();
      const auto warming = Clock::now();
      (void)run(runtime, benchmark::kTranscriptPrompts[0], 16, pages);
      const double warmup = std::chrono::duration<double>(Clock::now() - warming).count();
      std::cout << std::setprecision(12) << "{\"dictionary\":" << (dictionary ? "true" : "false")
                << ",\"gpu_coefficient_audit\":" << (audit ? "true" : "false")
                << ",\"dictionary_modules\":" << modules << ",\"metadata_units\":" << units
                << ",\"summed_per_projection_unique_pairs\":" << uniquePairs
                << ",\"additional_payload_bytes\":" << payload << ",\"conversion_s\":" << conversion
                << ",\"preparation_s\":" << preparation << ",\"warmup_s\":" << warmup
                << ",\"scope\":\"Native MLP AF4G64 staged-decode prototype; original metadata retained. "
                   "Additional payload is not a RAM saving. Other consumers and unsupported plans use original metadata. "
                   "B1 greedy, GPU-only, INT8 KV/FP32 state.\",\"measurements\":[";
      for (size_t i = 0; i < prompts.size(); ++i) {
        const auto result = run(runtime, prompts[i], outputTokens, pages);
        if (i) std::cout << ',';
        std::cout << "{\"input_tokens\":";
        array<uint32_t>(prompts[i]);
        std::cout << ",\"output_tokens\":";
        array<uint32_t>(result.output);
        std::cout << ",\"wall_s\":" << result.wall << ",\"ttft_s\":" << result.ttft << ",\"cycles\":[";
        for (size_t j = 0; j < result.cycles.size(); ++j) {
          const auto &cycle = result.cycles[j];
          if (j) std::cout << ',';
          std::cout << "{\"physical_rows\":" << cycle.rows << ",\"emitted\":" << cycle.emitted
                    << ",\"accepted\":" << cycle.accepted << ",\"wall_s\":" << cycle.wall
                    << ",\"gpu_s\":" << cycle.gpu << '}';
        }
        std::cout << "]}";
      }
      std::cout << "]}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "affine-dictionary-experiment: " << error.what() << '\n';
      return 1;
    }
  }
}
