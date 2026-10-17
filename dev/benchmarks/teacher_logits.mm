// Teacher-forced next-token logits of the production runtime, for numerical
// parity checks against another engine fed the same token ids.
//
//   teacher-logits METALLIB MODEL_ROOT TOKENS OUTPUT --window END:ROWS [--window END:ROWS...]
//                  [--kv-format int8|bf16] [--state-format fp32|bf16]
//                  [--expert-memory BYTES]
//
// TOKENS holds little-endian uint32 token ids, read as they are: no BOS, no
// chat template. For each window a cold request prefills ids [0, END - ROWS)
// and then decodes ROWS positions teacher-forced through the production
// verify: a thinking budget of no tokens whose end is the true continuation
// makes each verify row select the next true id and replaces the draft's
// proposals with them, so every cycle retains all its rows (the runtime's
// thinking cutoff); a sentinel id that is not in the window closes the end.
// OUTPUT-END.f32 receives ROWS + 1 rows of the target's vocabulary of fp32
// logits for positions END - ROWS - 1 .. END - 1, position p's predicting
// id p + 1: the row of the prefill's final position, then the verify rows,
// which RuntimeInstrumentation::lastLogits reads after each command.

#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "ops/PageStorage.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr uint32_t kRows = model::ExecutionLimits::targetVerifyRows;

struct Window final {
  uint32_t end = 0;
  uint32_t rows = 0;
};

uint64_t parseNumber(std::string_view text, std::string_view label) {
  uint64_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !value)
    throw std::invalid_argument(std::string(label) + " must be a positive integer");
  return value;
}

// END:ROWS, ROWS a positive multiple of the verify rows below END.
Window parseWindow(std::string_view text) {
  const size_t colon = text.find(':');
  if (colon == std::string_view::npos)
    throw std::invalid_argument("--window takes END:ROWS");
  const Window window{static_cast<uint32_t>(parseNumber(text.substr(0, colon), "a window's END")),
                      static_cast<uint32_t>(parseNumber(text.substr(colon + 1), "a window's ROWS"))};
  if (window.rows % kRows || window.rows >= window.end)
    throw std::invalid_argument("a window's ROWS must be a multiple of " + std::to_string(kRows) +
                                " below its END");
  return window;
}

std::vector<uint32_t> readTokens(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
  if (!file || bytes.size() % sizeof(uint32_t))
    throw std::runtime_error("cannot read uint32 token ids from " + path.string());
  std::vector<uint32_t> tokens(bytes.size() / sizeof(uint32_t));
  std::memcpy(tokens.data(), bytes.data(), bytes.size());
  return tokens;
}

// The logits a window collects, and how well they predict the true ids.
class WindowLogits final {
public:
  WindowLogits(uint32_t rows, uint32_t vocabulary)
      : vocabulary_(vocabulary), logits_(uint64_t{rows} * vocabulary) {}

  void put(uint32_t row, std::span<const float> logits, uint32_t next) {
    std::copy(logits.begin(), logits.end(), logits_.begin() + uint64_t{row} * vocabulary_);
    const auto top = std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float logit : logits)
      sum += std::exp(double(logit) - *top);
    negativeLogLikelihood_ += std::log(sum) + *top - logits[next];
    correct_ += static_cast<uint32_t>(top - logits.begin()) == next;
    ++filled_;
  }

  void write(const std::filesystem::path &path) const {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(logits_.data()),
               static_cast<std::streamsize>(logits_.size() * sizeof(float)));
    if (!file)
      throw std::runtime_error("cannot write " + path.string());
  }

  [[nodiscard]] double meanNegativeLogLikelihood() const { return negativeLogLikelihood_ / filled_; }
  [[nodiscard]] double accuracy() const { return double(correct_) / filled_; }

private:
  uint32_t vocabulary_;
  std::vector<float> logits_;
  double negativeLogLikelihood_ = 0.0;
  uint32_t correct_ = 0;
  uint32_t filled_ = 0;
};

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

// One window on state lane 0, which the request leaves free.
void runWindow(model::Runtime &executor, std::span<const uint32_t> tokens, Window window, uint64_t id,
               std::span<const uint32_t> pages, uint32_t vocabulary, const std::string &output) {
  const uint32_t start = window.end - window.rows;
  const std::span<const uint32_t> prompt = tokens.first(start);
  const std::span<const uint32_t> forced = tokens.subspan(start, window.rows + 1);
  uint32_t sentinel = vocabulary - 1;
  while (std::ranges::find(forced, sentinel) != forced.end())
    --sentinel;
  EngineRequest request;
  request.id = id;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = window.rows + 2 * kRows;
  ThinkingBudget budget{0, {forced.begin(), forced.end()}, sentinel};
  budget.end.push_back(sentinel);
  require(!budget.unusable(), "the forced ids cannot form a thinking end");
  request.thinkingBudget = std::move(budget);
  executor.beginColdRequest(request.modelView(), 0);

  WindowLogits logits(window.rows + 1, vocabulary);
  const auto started = std::chrono::steady_clock::now();
  const uint32_t chunk = executor.prefillGroupRows();
  for (uint32_t offset = 0; offset < start;) {
    const uint32_t count = std::min(chunk, start - offset);
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{id, count, offset}}, .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{id, offset, count, pages, 1};
    item.inputTokens = prompt.subspan(offset, count);
    const std::vector<ModelStepResult> results = executor.prefill(plan, std::span(&item, 1));
    require(results.size() == 1 && results[0].failure.empty() && results[0].consumedPromptTokens == count,
            "prefill failed: " + (results.empty() ? std::string("no result") : results[0].failure));
    offset += count;
    if (offset == start)
      require(results[0].firstToken == forced[0], "the prefill did not select the first forced id");
  }
  logits.put(0, model::RuntimeInstrumentation::lastLogits(executor, 0).first(vocabulary), forced[0]);
  const double prefillSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  uint32_t position = start;
  for (uint32_t cycle = 0; cycle < window.rows / kRows; ++cycle) {
    BatchPlan plan{.kind = WorkKind::Decode, .items = {{id, 0}}, .decodeStage = DecodeStage::Regular};
    const ModelBatchItem item{id, position, 0, pages, 1};
    const std::vector<ModelStepResult> results = executor.decode(plan, std::span(&item, 1));
    require(results.size() == 1 && results[0].failure.empty(),
            "decode failed: " + (results.empty() ? std::string("no result") : results[0].failure));
    // A cycle emits its anchor and the ids its rows retained before the next
    // anchor: every row's, when the forcing held.
    const std::vector<uint32_t> &emitted = results[0].outputTokens;
    require(emitted.size() == kRows && results[0].outputTokensWithoutKv == 0 &&
                std::ranges::equal(emitted, tokens.subspan(position, kRows)),
            "decode cycle " + std::to_string(cycle) + " did not retain the forced ids");
    const std::span<const float> rows = model::RuntimeInstrumentation::lastLogits(executor, 0);
    for (uint32_t row = 0; row < kRows; ++row)
      logits.put(1 + cycle * kRows + row, rows.subspan(uint64_t{row} * vocabulary, vocabulary),
                 tokens[position + row + 1]);
    position += kRows;
  }
  executor.end(id);
  const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const std::string path = output + "-" + std::to_string(window.end) + ".f32";
  logits.write(path);
  std::printf("window end=%u rows=%u prefill_s=%.1f decode_s=%.1f nll=%.4f top1_true=%.4f -> %s\n", window.end,
              window.rows + 1, prefillSeconds, totalSeconds - prefillSeconds, logits.meanNegativeLogLikelihood(),
              logits.accuracy(), path.c_str());
  std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 5) {
        std::cerr << "usage: teacher-logits METALLIB MODEL_ROOT TOKENS OUTPUT --window END:ROWS "
                     "[--window END:ROWS...] [--kv-format int8|bf16] [--state-format fp32|bf16] "
                     "[--expert-memory BYTES]\n";
        return 2;
      }
      std::vector<Window> windows;
      kv::Format format = kv::Format::Int8;
      ops::GdnStateFormat stateFormat = ops::GdnStateFormat::Float32;
      model::ModelLoadOptions loadOptions;
      for (int index = 5; index < argc; index += 2) {
        const std::string_view option(argv[index]);
        if (index + 1 >= argc)
          throw std::invalid_argument(std::string(option) + " requires a value");
        const std::string_view value(argv[index + 1]);
        if (option == "--window")
          windows.push_back(parseWindow(value));
        else if (option == "--kv-format") {
          if (value != "int8" && value != "bf16")
            throw std::invalid_argument("--kv-format takes int8 or bf16");
          format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
        } else if (option == "--state-format") {
          if (value != "fp32" && value != "bf16")
            throw std::invalid_argument("--state-format takes fp32 or bf16");
          stateFormat = value == "fp32" ? ops::GdnStateFormat::Float32 : ops::GdnStateFormat::BFloat16;
        } else if (option == "--expert-memory")
          loadOptions.expertMemoryBytes = parseNumber(value, "--expert-memory");
        else
          throw std::invalid_argument("unknown option " + std::string(option));
      }
      if (windows.empty())
        throw std::invalid_argument("no --window given");
      const std::vector<uint32_t> tokens = readTokens(argv[3]);
      uint32_t lastEnd = 0;
      for (const Window &window : windows)
        lastEnd = std::max(lastEnd, window.end);
      // The last window's last row predicts id END.
      require(lastEnd < tokens.size(), "TOKENS holds no id " + std::to_string(lastEnd));

      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path root(argv[2]);
      model::LoadedModel model = model::loadModel(backend, root, model::inspectModelRoot(root), loadOptions);
      ops::ExecutionPlans operators(backend.capabilities());
      const uint32_t vocabulary =
          std::visit([](const auto &layout) { return layout.vocabularySize; }, model.descriptor.target);
      for (const uint32_t token : tokens)
        require(token < vocabulary, "TOKENS holds an id outside the vocabulary");

      // The longest window's rows and its last cycle's, in whole extents.
      const kv::Layout kvLayout = model.targetKvLayout(format);
      const uint32_t extentPages = kvLayout.maximumExtentPages();
      const uint32_t lanePages = (lastEnd + kRows + kv::kPageTokens - 1) / kv::kPageTokens;
      const uint32_t pageCount = (lanePages + extentPages - 1) / extentPages * extentPages;
      MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1, queryHostMemory,
                              0);
      kv::PageStorage pages(backend, governor.allocationAdmission(), kvLayout, pageCount, extentPages);
      for (uint32_t extent = 0; extent < pageCount / extentPages; ++extent)
        require(static_cast<bool>(pages.allocateExtent(extent)), "could not allocate the KV extents");
      model::QwenStateStorage states(backend, governor.allocationAdmission(), model.stateLayout(stateFormat),
                                     nullptr);
      model::RuntimeContext context{backend, model, pages, states, operators};
      model::Runtime executor(context);

      std::vector<uint32_t> pageTable(lanePages);
      for (uint32_t page = 0; page < lanePages; ++page)
        pageTable[page] = page;
      std::printf("device %s, %s KV, %s state, %zu ids\n", backend.capabilities().deviceName.c_str(),
                  format == kv::Format::Int8 ? "int8" : "bf16", std::string(ops::formatName(stateFormat)).c_str(),
                  tokens.size());
      for (size_t index = 0; index < windows.size(); ++index)
        runWindow(executor, tokens, windows[index], index + 1, pageTable, vocabulary, argv[4]);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "teacher-logits: " << error.what() << '\n';
      return 1;
    }
  }
}
