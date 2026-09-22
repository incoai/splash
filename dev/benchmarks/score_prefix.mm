// Teacher-forced scoring for parity checks: for each job (token prefix +
// 2..255 candidate token ids) runs a score-only prefill and prints the raw
// final-position logits at the candidates. Jobs come from stdin, one per line:
//   <prefix ids, comma separated>\t<candidate ids, comma separated>
// Output, one line per job: the candidate logits, comma separated.
//   make benchmark-score-prefix MODEL_ROOT=... < jobs.tsv > logits.txt
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "ops/Q8PageStorage.hpp"

#include <charconv>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

std::vector<uint32_t> parseList(std::string_view text) {
  std::vector<uint32_t> values;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    uint32_t value = 0;
    auto result = std::from_chars(item.data(), item.data() + item.size(), value);
    if (result.ec != std::errc{} || result.ptr != item.data() + item.size())
      throw std::invalid_argument("bad token id: " + std::string(item));
    values.push_back(value);
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
  }
  return values;
}

std::vector<uint32_t> pageRange(uint32_t count) {
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index) result[index] = index;
  return result;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 3) {
        std::cerr << "usage: score-prefix METALLIB MODEL_ROOT < jobs.tsv\n";
        return 2;
      }
      // Read every job first so the prompt length bound is known before the
      // KV pool is sized.
      struct Job { std::vector<uint32_t> prefix, candidates; };
      std::vector<Job> jobs;
      std::string line;
      uint32_t longest = 0;
      while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) throw std::invalid_argument("job line has no tab");
        Job job{parseList(std::string_view(line).substr(0, tab)),
                parseList(std::string_view(line).substr(tab + 1))};
        if (job.prefix.empty() || job.candidates.size() < 2 || job.candidates.size() > 255)
          throw std::invalid_argument("a job needs a prefix and 2..255 candidates");
        longest = std::max<uint32_t>(longest, job.prefix.size());
        jobs.push_back(std::move(job));
      }
      if (jobs.empty()) throw std::invalid_argument("no jobs on stdin");

      metal::MetalBackend backend(argv[1]);
      model::ModelPackage model =
          model::loadModelPackage(backend, std::filesystem::path(argv[2]));
      ops::ExecutionPlans operators(backend.capabilities());
      model::ModelMemoryPlan executorPlan =
          model::plannedRuntimeMemory(backend.capabilities(), model, operators);
      const uint32_t pagesNeeded =
          (longest + model::ExecutionLimits::targetVerifyRows) / kv::kPageTokens + 2;
      const uint32_t batchPages = model.targetKvLayout().sparseMappingBatchPages();
      const uint32_t pageCount = (pagesNeeded + batchPages - 1) / batchPages * batchPages;
      MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
      kv::Q8PageStorage pages(backend, governor.allocationAdmission(),
                              model.targetKvLayout(), pageCount);
      for (uint32_t page = 0; page < pageCount; ++page)
        if (!pages.ensureResident(page)) throw std::runtime_error("could not back the KV pages");
      model::QwenStateStorage states(backend, governor.allocationAdmission(),
                                     model.stateLayout());
      model::RuntimeContext context{backend, governor.allocationAdmission(), model, pages,
                                    states, operators, 16384,
                                    executorPlan.pipelineReserveBytes,
                                    executorPlan.runtimeOverheadReserveBytes};
      model::Runtime executor(context);
      const std::vector<uint32_t> pageTable = pageRange(pageCount);
      std::fprintf(stderr, "score-prefix: %s, %zu jobs, longest prefix %u\n",
                   backend.capabilities().deviceName.c_str(), jobs.size(), longest);

      const auto started = std::chrono::steady_clock::now();
      uint64_t requestId = 1;
      for (const Job &job : jobs) {
        EngineRequest request;
        request.id = requestId;
        request.prompt = job.prefix;
        request.maxNewTokens = 0;
        request.scoreTokens = job.candidates;
        executor.beginColdRequest(request.modelView(), 0);
        uint32_t offset = 0;
        std::vector<float> logits;
        while (offset < job.prefix.size()) {
          const uint32_t count = std::min<uint32_t>(
              model::ExecutionLimits::prefillTokenBudget,
              static_cast<uint32_t>(job.prefix.size()) - offset);
          BatchPlan plan{WorkKind::Prefill, BatchCohort::Greedy,
                         {{requestId, count, offset}}, DecodeStage::Regular};
          ModelBatchItem item{requestId, 0, offset, offset, count, pageTable};
          item.inputTokens = std::span<const uint32_t>(job.prefix).subspan(offset, count);
          auto results = executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
          if (results.size() != 1 || results[0].consumedPromptTokens != count)
            throw std::runtime_error("prefill consumed the wrong row count");
          offset += count;
          if (offset == job.prefix.size()) logits = results[0].scoreLogits;
        }
        executor.end(requestId);
        ++requestId;
        if (logits.size() != job.candidates.size())
          throw std::runtime_error("score prefill returned " + std::to_string(logits.size()) +
                                   " logits for " + std::to_string(job.candidates.size()) +
                                   " candidates");
        std::string out;
        for (size_t i = 0; i < logits.size(); ++i) {
          char buffer[32];
          std::snprintf(buffer, sizeof buffer, "%.6g", logits[i]);
          out += (i ? "," : "") + std::string(buffer);
        }
        std::puts(out.c_str());
        if (requestId % 100 == 0) {
          const double seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
          std::fprintf(stderr, "  %llu jobs, %.1f s\n",
                       static_cast<unsigned long long>(requestId - 1), seconds);
        }
      }
      std::fflush(stdout);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "score-prefix: " << error.what() << '\n';
      return 1;
    }
  }
}
