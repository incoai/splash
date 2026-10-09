// Times Splash's production attention kernels on one layer of Page32 KV
// across cache lengths, for the prefill chunk (2048 rows), and the DFlash
// verify batch (8 rows per lane, one and four lanes). Each case builds the same
// store + attention graph the executor encodes, reports the fused GPU time of
// the whole graph and, with each dispatch submitted as its own command, the GPU
// time of each pipeline over the deterministic synthetic history of
// dev/tests/engine/AttentionFixture.hpp. These are kernel timings, not a
// correctness oracle (the attention kernel tests are). The KV sits in extents
// of the size the memory plan picks for the model, or of --extent-pages pages,
// which must hold whole alignment units of every swept shape; the swept layer
// is the second of two so that its region starts past the first one's, and each
// case prints a digest of its output: two builds that fill the same pages must
// print the same digests, whatever their storage.
//
// usage: attention-sweep METALLIB [--histories 0,2048,...] [--shapes 27b,35b]
//                        [--lanes 1,4] [--repeat N] [--phases both|verify|prefill]
//                        [--compare-metallib PATH] [--kv-format int8|bf16]
//                        [--extent-pages N]
//
// The comparison library loads into a MetalBackend of its own, which needs
// residency_kick (kernels/shared/residency.metal) in every library it loads:
// build baselines from a tree that has that kernel.
#include "../tests/engine/AttentionFixture.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <iomanip>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

using test::AttentionFixture;
using test::AttentionFixturePlan;
using test::AttentionShape;

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPrefillRows = SPLASH_PREFILL_TOKEN_BUDGET;

// The swept attention layer is the second of the extents' two.
constexpr uint32_t kLayer = 1;
// The memory plan's extents of the full models: 16 and 10 attention layers.
constexpr uint32_t kModelLayers27b = 16, kModelLayers35b = 10;

AttentionShape shapeOf(const std::string &shape, kv::Format format) {
  return shape == "27b" ? AttentionShape{24, 4, 256, format}
                        : AttentionShape{16, 2, 256, format};
}

// The prefill chunk on one lane, or the verify rows of `lanes` lanes, all
// after `history` tokens, with their plan's scratch.
AttentionFixturePlan casePlan(AttentionShape shape, bool prefill, uint32_t lanes,
                              uint32_t history, uint32_t extentPages) {
  const kv::Layout layout{1, shape.kvHeads, shape.headDimension, shape.format};
  const uint32_t caseLanes = prefill ? 1 : lanes;
  AttentionFixturePlan::Histories histories{};
  std::fill_n(histories.begin(), caseLanes, history);
  const AttentionWorkspace scratch =
      prefill ? PagedAttention::prefillPlan(kPrefillRows, shape.queryHeads, layout)
                    .workspace
              : PagedAttention::verifyPlan(caseLanes, shape.queryHeads, layout,
                                           std::span(histories).first(caseLanes))
                    .workspace;
  return AttentionFixturePlan::make(shape, caseLanes, prefill ? kPrefillRows : kVerifyRows,
                                    histories, scratch, {kLayer + 1, kLayer, extentPages});
}

// The store and attention graph the runtime encodes for a case.
metal::CommandGraph caseGraph(const AttentionFixture &fixture, bool prefill) {
  const AttentionFixturePlan &plan = fixture.plan();
  metal::CommandGraph graph;
  if (prefill)
    fixture.addGraph(graph, PagedAttention::prefillPlan(plan.rows, plan.shape.queryHeads,
                                                        plan.layout()));
  else
    fixture.addGraph(graph, PagedAttention::verifyPlan(
                                plan.lanes, plan.shape.queryHeads, plan.layout(),
                                std::span(plan.histories).first(plan.lanes)));
  return graph;
}

// Both cache payloads of every lane's history and rows, including scales
// only for INT8.
uint64_t historyBytes(const AttentionFixturePlan &plan) {
  uint64_t tokens = 0;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane)
    tokens += uint64_t{plan.histories[lane]} + plan.rows;
  return tokens * (plan.layout().bytesPerLayerPage() / kv::kPageTokens);
}

std::span<const uint8_t> outputBytes(const AttentionFixture &fixture) {
  const metal::MetalBuffer output = fixture.buffer(AttentionFixture::Tensor::Output);
  return {static_cast<const uint8_t *>(output.contents()), output.sizeBytes()};
}

// FNV-1a of the attention output.
uint64_t outputDigest(const AttentionFixture &fixture) {
  uint64_t digest = 0xcbf29ce484222325ULL;
  for (const uint8_t byte : outputBytes(fixture))
    digest = (digest ^ byte) * 0x100000001b3ULL;
  return digest;
}

struct Case final {
  double fusedMilliseconds = 0.0;
  std::map<std::string, double> pipelineMilliseconds;
  uint64_t kvBytes = 0;
  uint64_t outputDigest = 0;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// warmupSeconds of GPU time on every variant come first: the first case of
// a run also brings an idle GPU up to its clocks.
std::vector<Case> measure(std::span<metal::MetalBackend *> backends,
                          const AttentionFixturePlan &plan, bool prefill, uint32_t repeat,
                          double warmupSeconds) {
  std::vector<std::unique_ptr<AttentionFixture>> fixtures;
  std::vector<metal::CommandGraph> graphs;
  std::vector<Case> results(backends.size());
  for (size_t i = 0; i < backends.size(); ++i) {
    fixtures.push_back(
        std::make_unique<AttentionFixture>(*backends[i], plan, "attention-sweep-fixture"));
    fixtures.back()->fill();
    graphs.push_back(caseGraph(*fixtures.back(), prefill));
    results[i].kvBytes = historyBytes(plan);
  }
  // Warm every variant, then alternate order to limit clock/thermal drift.
  double warmup = 0.0;
  while (warmup < warmupSeconds)
    for (size_t i = 0; i < backends.size(); ++i)
      warmup += backends[i]->submitCommandAsync(graphs[i].dispatches()).wait().gpuSeconds;
  for (size_t i = 1; i < fixtures.size(); ++i)
    if (!std::ranges::equal(outputBytes(*fixtures[0]), outputBytes(*fixtures[i])))
      throw std::runtime_error("comparison metallib changed attention output bits");
  for (size_t i = 0; i < fixtures.size(); ++i)
    results[i].outputDigest = outputDigest(*fixtures[i]);
  std::vector<std::vector<double>> fused(backends.size());
  std::vector<std::map<std::string, std::vector<double>>> perPipeline(backends.size());
  for (uint32_t round = 0; round < repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      fused[i].push_back(backends[i]->submitCommandAsync(graphs[i].dispatches()).wait().gpuSeconds * 1000.0);
    }
  for (uint64_t round = 0; round <= repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      const auto run =
          benchmark::replayDispatches(*backends[i], graphs[i].dispatches());
      if (round)
        for (const auto &[name, seconds] : run)
          perPipeline[i][name].push_back(seconds * 1000.0);
    }
  for (size_t i = 0; i < backends.size(); ++i) {
    results[i].fusedMilliseconds = median(fused[i]);
    for (auto &[name, samples] : perPipeline[i])
      results[i].pipelineMilliseconds[name] = median(samples);
  }
  return results;
}

uint32_t parseCount(std::string_view text, uint32_t minimum, uint32_t maximum,
                    std::string_view option) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value < minimum || value > maximum)
    throw std::invalid_argument(std::string(option) + " requires integers from " +
                                std::to_string(minimum) + " to " + std::to_string(maximum));
  return value;
}

std::vector<uint32_t> parseList(const std::string &text, uint32_t minimum,
                                uint32_t maximum, std::string_view option) {
  std::vector<uint32_t> values;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    const std::string item = text.substr(start, comma == std::string::npos ? std::string::npos
                                                                            : comma - start);
    values.push_back(parseCount(item, minimum, maximum, option));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return values;
}

std::string hex(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::string json(const Case &item, const std::string &shape, uint32_t history,
                 const std::string &kind, uint32_t lanes, size_t variant) {
  std::string out = "{\"variant\":" + std::to_string(variant) + ",\"shape\":\"" + shape + "\",\"history\":" + std::to_string(history) +
                    ",\"kind\":\"" + kind + "\",\"lanes\":" + std::to_string(lanes) +
                    ",\"fused_ms\":" + std::to_string(item.fusedMilliseconds) +
                    ",\"kv_bytes\":" + std::to_string(item.kvBytes) +
                    ",\"output_digest\":\"" + hex(item.outputDigest) + "\",\"pipelines\":{";
  bool first = true;
  for (const auto &[name, milliseconds] : item.pipelineMilliseconds) {
    out += (first ? "" : ",") + std::string("\"") + name + "\":" + std::to_string(milliseconds);
    first = false;
  }
  return out + "}}";
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "usage: attention-sweep METALLIB [--histories LIST] [--shapes 27b,35b] "
                   "[--lanes LIST] [--repeat N] [--phases both|verify|prefill] "
                   "[--compare-metallib PATH] [--kv-format int8|bf16] "
                   "[--extent-pages N]\n";
      return 64;
    }
    std::vector<uint32_t> histories{0, 2048, 8192, 16384, 32768, 65536, 131072};
    std::vector<uint32_t> lanes{1, 4};
    std::vector<std::string> shapes{"27b", "35b"};
    uint32_t repeat = 5;
    // Zero: the largest extent the memory plan picks for the model.
    uint32_t extentPages = 0;
    kv::Format format = kv::Format::Int8;
    std::string comparisonLibrary, phases = "both";
    for (int index = 2; index < argc; index += 2) {
      const std::string option(argv[index]);
      if (index + 1 >= argc)
        throw std::invalid_argument(option + " requires a value");
      if (option == "--histories")
        histories = parseList(argv[index + 1], 0,
                              kv::kMaximumPhysicalTokens - kPrefillRows, option);
      else if (option == "--lanes")
        lanes = parseList(argv[index + 1], 1, kMaximumLanes, option);
      else if (option == "--repeat")
        repeat = parseCount(argv[index + 1], 1, std::numeric_limits<uint32_t>::max(), option);
      else if (option == "--extent-pages")
        extentPages = parseCount(argv[index + 1], 1, SPLASH_KV_PAGE_INDEX_MASK, option);
      else if (option == "--kv-format") {
        const std::string_view value(argv[index + 1]);
        if (value != "int8" && value != "bf16")
          throw std::invalid_argument("--kv-format takes int8 or bf16");
        format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
      }
      else if (option == "--compare-metallib") comparisonLibrary = argv[index + 1];
      else if (option == "--phases") {
        phases = argv[index + 1];
        if (phases != "both" && phases != "verify" && phases != "prefill")
          throw std::invalid_argument("--phases takes both, verify or prefill");
      }
      else if (option == "--shapes") {
        shapes.clear();
        std::string text(argv[index + 1]);
        size_t start = 0;
        while (start <= text.size()) {
          const size_t comma = text.find(',', start);
          const std::string shape =
              text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
          if (shape != "27b" && shape != "35b")
            throw std::invalid_argument("--shapes takes 27b or 35b");
          shapes.push_back(shape);
          if (comma == std::string::npos) break;
          start = comma + 1;
        }
      } else throw std::invalid_argument("unknown option " + option);
    }
    for (const std::string &shape : shapes) {
      const AttentionShape geometry = shapeOf(shape, format);
      if (extentPages % kv::Layout{1, geometry.kvHeads, geometry.headDimension, format}
                            .extentAlignmentPages())
        throw std::invalid_argument("--extent-pages " + std::to_string(extentPages) +
                                    " does not hold whole alignment units of " + shape +
                                    " " + std::string(kv::formatName(format)));
    }
    metal::MetalBackend backend(argv[1]);
    std::unique_ptr<metal::MetalBackend> comparison;
    std::vector<metal::MetalBackend *> backends{&backend};
    if (!comparisonLibrary.empty()) {
      comparison = std::make_unique<metal::MetalBackend>(comparisonLibrary);
      backends.push_back(comparison.get());
    }
    std::cerr << "device " << backend.capabilities().deviceName << ", one attention layer, "
              << "Page32 " << kv::formatName(format) << " KV, median of " << repeat << " fused graphs (ms)\n";
    double warmupSeconds = 1.5;
    std::cout << "{\"device\":\"" << backend.capabilities().deviceName << "\",\"kv_format\":\"" << kv::formatName(format) << "\",\"cases\":[";
    bool firstCase = true;
    for (const std::string &shape : shapes) {
      const AttentionShape geometry = shapeOf(shape, format);
      const std::string name = shape == "27b" ? "qwen3.8-27b" : "qwen3.6-35b-a3b";
      const uint32_t shapeExtentPages =
          extentPages ? extentPages
                      : kv::Layout{shape == "27b" ? kModelLayers27b : kModelLayers35b,
                                   geometry.kvHeads, geometry.headDimension, format}
                            .maximumExtentPages();
      std::cerr << "\n" << name << "  (" << geometry.queryHeads << " query heads, "
                << geometry.kvHeads << " KV heads, d=" << geometry.headDimension
                << ", extents of " << shapeExtentPages << " pages)\n";
      for (uint32_t history : histories) {
        auto report = [&](bool prefill, uint32_t lane) {
          const auto cases =
              measure(backends, casePlan(geometry, prefill, lane, history, shapeExtentPages),
                      prefill, repeat, warmupSeconds);
          warmupSeconds = 0.1;
          for (size_t i = 0; i < cases.size(); ++i) {
            std::cout << (firstCase ? "" : ",")
                      << json(cases[i], name, history, prefill ? "prefill" : "verify", lane, i);
            firstCase = false;
            std::cerr << history << " " << (prefill ? "prefill" : "verify")
                      << " lanes=" << lane << " variant=" << i << " fused="
                      << cases[i].fusedMilliseconds << " ms digest="
                      << hex(cases[i].outputDigest) << "\n";
          }
        };
        if (phases != "verify") report(true, 1);
        if (phases != "prefill") for (uint32_t lane : lanes) report(false, lane);
      }
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "attention-sweep: " << error.what() << '\n';
    return 70;
  }
}
