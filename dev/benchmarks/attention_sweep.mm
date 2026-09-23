// Times Splash's production attention kernels on one layer of Page32 Q8 KV
// across cache lengths, for the prefill chunk (2048 rows), and the DFlash
// verify batch (8 rows per lane, one and four lanes). Each case builds the same
// store + attention graph the executor encodes, reports the fused GPU time of
// the whole graph and, with dispatch profiling, the GPU time of each pipeline
// over deterministic synthetic history. These are kernel
// timings, not a correctness oracle (the tuning tests are).
//
// usage: attention-sweep METALLIB [--histories 0,2048,...] [--shapes 27b,35b]
//                        [--lanes 1,4] [--repeat N] [--phases both|verify|prefill]
//                        [--compare-metallib PATH] [--kv-format int8|bf16]
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

constexpr uint64_t kAlignment = 16 * 1024;
constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDimension = 256;
constexpr uint32_t kPageRows = kv::kPageTokens;
constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPrefillRows = SPLASH_PREFILL_TOKEN_BUDGET;

enum class Tensor : uint32_t {
  Keys, KeyScales, Values, ValueScales, ChunkKeys, ChunkValues,
  Queries, Output, Partials, Statistics, Table0, Table1, Table2, Table3, Count
};
constexpr size_t tensorIndex(Tensor tensor) { return static_cast<size_t>(tensor); }
constexpr size_t kTensorCount = tensorIndex(Tensor::Count);

uint64_t aligned(uint64_t bytes) {
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

struct Plan final {
  AttentionShape shape;
  bool prefill = false;
  VerifyAttentionConfig verify;
  uint32_t lanes = 0;
  uint32_t rows = 0;
  uint32_t stride = 0;
  uint32_t physicalPages = 0;
  std::array<uint32_t, kMaximumLanes> histories{};
  std::array<uint32_t, kMaximumLanes> pages{};
  std::array<uint64_t, kTensorCount> sizes{};
  uint64_t bytes = 0;

  kv::Layout layout() const { return {1, shape.kvHeads, shape.headDimension, shape.format}; }
  // The verify plan scales each lane's split count with its own history.
  std::span<const uint32_t> laneHistories() const { return {histories.data(), lanes}; }
  void size(Tensor tensor, uint64_t value) { sizes[tensorIndex(tensor)] = value; }
};

Plan makePlan(AttentionShape shape, bool prefill, uint32_t lanes, uint32_t history,
              VerifyAttentionConfig verify = {}) {
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("--lanes must contain values from 1 to 4");
  Plan plan;
  plan.shape = shape;
  plan.prefill = prefill;
  plan.verify = verify;
  plan.lanes = prefill ? 1 : lanes;
  plan.rows = prefill ? kPrefillRows : kVerifyRows;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane) plan.histories[lane] = history;
  AttentionWorkspace scratch;
  if (prefill) {
    scratch = PagedAttention::prefillPlan(plan.rows, shape.queryHeads, plan.layout(),
                                          history, PrefillAttentionConfig{})
                  .workspace;
  } else {
    scratch = PagedAttention::verifyPlan(plan.lanes, shape.queryHeads, plan.layout(),
                                         plan.laneHistories(), verify)
                  .workspace;
  }
  plan.stride = (plan.rows + kPageRows - 1) / kPageRows * kPageRows;
  uint32_t pages = 0;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane) {
    const uint64_t tokens = uint64_t{plan.histories[lane]} + plan.rows;
    if (tokens > kv::kMaximumPhysicalTokens)
      throw std::invalid_argument("history exceeds the KV context");
    plan.pages[lane] = uint32_t((tokens + kPageRows - 1) / kPageRows);
    pages += plan.pages[lane];
    plan.size(static_cast<Tensor>(tensorIndex(Tensor::Table0) + lane),
              uint64_t{plan.pages[lane]} * sizeof(uint32_t));
  }
  plan.physicalPages = pages + 1 + (pages % 2);
  plan.size(Tensor::Keys, plan.physicalPages * plan.layout().dataBytesPerLayerPage());
  plan.size(Tensor::Values, plan.physicalPages * plan.layout().dataBytesPerLayerPage());
  plan.size(Tensor::KeyScales, plan.physicalPages * plan.layout().scaleBytesPerLayerPage());
  plan.size(Tensor::ValueScales, plan.physicalPages * plan.layout().scaleBytesPerLayerPage());
  const uint64_t chunks = uint64_t{plan.lanes} * shape.kvHeads * plan.stride * kDimension * 2;
  const uint64_t queries = uint64_t{plan.lanes} * shape.queryHeads * plan.stride * kDimension * 2;
  plan.size(Tensor::ChunkKeys, chunks);
  plan.size(Tensor::ChunkValues, chunks);
  plan.size(Tensor::Queries, queries);
  plan.size(Tensor::Output, queries);
  plan.size(Tensor::Partials, scratch.partialsBytes);
  plan.size(Tensor::Statistics, scratch.statisticsBytes);
  for (uint64_t size : plan.sizes) plan.bytes += aligned(size);
  return plan;
}

class Fixture final {
public:
  Fixture(metal::MetalBackend &backend, Plan plan)
      : plan_(std::move(plan)),
        base_(backend.allocateBuffer(plan_.bytes, metal::BufferStorage::Shared,
                                     "attention-sweep-fixture")) {
    uint64_t offset = 0;
    for (size_t i = 0; i < plan_.sizes.size(); ++i) {
      if (plan_.sizes[i]) buffers_[i] = backend.view(base_, offset, plan_.sizes[i]);
      offset += aligned(plan_.sizes[i]);
    }
    layer_ = {get(Tensor::Keys), get(Tensor::KeyScales), get(Tensor::Values),
              get(Tensor::ValueScales), plan_.shape.format};
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane) {
      tables_[lane] = get(static_cast<Tensor>(tensorIndex(Tensor::Table0) + lane));
      stores_[lane] = {plan_.histories[lane], plan_.rows, plan_.stride, plan_.pages[lane],
                       plan_.physicalPages, 0, 0, 0};
      attention_[lane] = kv::q8VerifyAttentionParams(
          plan_.histories[lane], kVerifyRows, plan_.stride, plan_.pages[lane],
          plan_.physicalPages);
    }
    for (uint32_t lane = plan_.lanes; lane < kMaximumLanes; ++lane) {
      tables_[lane] = tables_[0];
      stores_[lane] = stores_[0];
      attention_[lane] = attention_[0];
    }
    initialize();
  }

  metal::CommandGraph graph() const {
    metal::CommandGraph result;
    if (plan_.prefill) {
      const auto attentionPlan = PagedAttention::prefillPlan(
          plan_.rows, plan_.shape.queryHeads, plan_.layout(), plan_.histories[0],
          PrefillAttentionConfig{});
      PagedAttention::addPrefillStore(result, layer_, get(Tensor::ChunkKeys),
                                      get(Tensor::ChunkValues), tables_[0], stores_[0],
                                      plan_.layout());
      PagedAttention::addPrefill(result, layer_, get(Tensor::Queries), get(Tensor::Output),
                                 get(Tensor::Partials), get(Tensor::Statistics), tables_[0],
                                 stores_[0], attentionPlan);
    } else {
      const auto attentionPlan = PagedAttention::verifyPlan(
          plan_.lanes, plan_.shape.queryHeads, plan_.layout(), plan_.laneHistories(),
          plan_.verify);
      PagedAttention::addVerify(
          result, layer_,
          {get(Tensor::ChunkKeys), get(Tensor::ChunkValues), get(Tensor::Queries),
           get(Tensor::Partials), get(Tensor::Statistics), get(Tensor::Output), tables_},
          stores_, attention_, attentionPlan);
    }
    return result;
  }

  bool sameOutput(const Fixture &other) const {
    const auto left = get(Tensor::Output), right = other.get(Tensor::Output);
    return left.sizeBytes() == right.sizeBytes() &&
           std::memcmp(left.contents(), right.contents(), left.sizeBytes()) == 0;
  }

  uint64_t historyBytes() const {
    // Both cache payloads, including scales only for INT8.
    uint64_t tokens = 0;
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane)
      tokens += uint64_t{plan_.histories[lane]} + plan_.rows;
    return tokens * (plan_.layout().bytesPerLayerPage() / kPageRows);
  }

private:
  void initialize() {
    std::memset(base_.contents(), 0, plan_.bytes);
    auto *keys = static_cast<int8_t *>(get(Tensor::Keys).contents());
    auto *values = static_cast<int8_t *>(get(Tensor::Values).contents());
    auto *keyScales = static_cast<float *>(get(Tensor::KeyScales).contents());
    auto *valueScales = static_cast<float *>(get(Tensor::ValueScales).contents());
    auto *chunkKeys = static_cast<uint16_t *>(get(Tensor::ChunkKeys).contents());
    auto *chunkValues = static_cast<uint16_t *>(get(Tensor::ChunkValues).contents());
    auto *queries = static_cast<uint16_t *>(get(Tensor::Queries).contents());
    const uint32_t group = plan_.shape.queryHeads / plan_.shape.kvHeads;
    uint32_t firstPage = 0;
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane) {
      auto *table = static_cast<uint32_t *>(tables_[lane].contents());
      for (uint32_t page = 0; page < plan_.pages[lane]; ++page)
        table[page] = (2 * (firstPage + page) + 1) % plan_.physicalPages;
      firstPage += plan_.pages[lane];
      for (uint32_t token = 0; token < plan_.histories[lane]; ++token) {
        for (uint32_t head = 0; head < plan_.shape.kvHeads; ++head) {
          const uint64_t scale =
              (uint64_t{table[token / kPageRows]} * plan_.shape.kvHeads + head) * kPageRows +
              token % kPageRows;
          if (plan_.shape.format == kv::Format::Int8) {
            keyScales[scale] = 0.006f;
            valueScales[scale] = 0.007f;
          }
          for (uint32_t d = 0; d < kDimension; ++d) {
            const int key = int((uint64_t{token} * 37 + head * 101 + d * 17 + lane * 7) % 255) - 127;
            const int value = int((uint64_t{token} * 53 + head * 79 + d * 29 + lane * 19) % 255) - 127;
            const uint64_t ki = scale * kDimension + d;
            const uint64_t vi = (scale / kPageRows * kDimension + d) * kPageRows + scale % kPageRows;
            if (plan_.shape.format == kv::Format::Int8) {
              keys[ki] = key;
              values[vi] = value;
            } else {
              static_cast<uint16_t *>(get(Tensor::Keys).contents())[ki] = bf16(key * 0.006f);
              static_cast<uint16_t *>(get(Tensor::Values).contents())[vi] = bf16(value * 0.007f);
            }
          }
        }
      }
      for (uint32_t row = 0; row < plan_.rows; ++row) {
        for (uint32_t head = 0; head < plan_.shape.kvHeads; ++head) {
          const uint64_t base =
              (uint64_t{lane} * plan_.shape.kvHeads + head) * plan_.stride * kDimension;
          for (uint32_t d = 0; d < kDimension; ++d) {
            chunkKeys[base + row * kDimension + d] =
                bf16(float(int((row * 37 + head * 101 + d * 17 + lane * 7) % 255) - 127) * 0.006f);
            chunkValues[base + d * plan_.stride + row] =
                bf16(float(int((row * 53 + head * 79 + d * 29 + lane * 19) % 255) - 127) * 0.007f);
          }
        }
        for (uint32_t head = 0; head < plan_.shape.queryHeads; ++head)
          for (uint32_t d = 0; d < kDimension; ++d) {
            const uint64_t index =
                (((uint64_t{lane} * plan_.shape.kvHeads + head / group) * plan_.stride + row) *
                     group +
                 head % group) *
                    kDimension +
                d;
            queries[index] = bf16(
                float(int((row * 43 + head * 67 + d * 11 + head * d * 7 + lane * 29) % 1019) -
                      509) /
                1018.0f);
          }
      }
    }
  }

  metal::MetalBuffer get(Tensor tensor) const { return buffers_[tensorIndex(tensor)]; }

  Plan plan_;
  metal::MetalBuffer base_;
  std::array<metal::MetalBuffer, kTensorCount> buffers_{};
  kv::LayerStorage layer_;
  std::array<metal::MetalBuffer, kMaximumLanes> tables_{};
  std::array<kv::Q8ChunkedPrefillParams, kMaximumLanes> stores_{};
  std::array<kv::Q8VerifyAttentionParams, kMaximumLanes> attention_{};
};

struct Case final {
  double fusedMilliseconds = 0.0;
  std::map<std::string, double> pipelineMilliseconds;
  uint64_t kvBytes = 0;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::vector<Case> measure(std::span<metal::MetalBackend *> backends,
                          const Plan &plan, uint32_t repeat) {
  std::vector<std::unique_ptr<Fixture>> fixtures;
  std::vector<metal::CommandGraph> graphs;
  std::vector<Case> results(backends.size());
  for (size_t i = 0; i < backends.size(); ++i) {
    fixtures.push_back(std::make_unique<Fixture>(*backends[i], plan));
    graphs.push_back(fixtures.back()->graph());
    results[i].kvBytes = fixtures.back()->historyBytes();
  }
  // Warm every variant, then alternate order to limit clock/thermal drift.
  double warmup = 0.0;
  while (warmup < 0.1)
    for (size_t i = 0; i < backends.size(); ++i)
      warmup += backends[i]->submitCommand(graphs[i].dispatches()).gpuSeconds;
  for (size_t i = 1; i < fixtures.size(); ++i)
    if (!fixtures[0]->sameOutput(*fixtures[i]))
      throw std::runtime_error("comparison metallib changed attention output bits");
  std::vector<std::vector<double>> fused(backends.size());
  std::vector<std::map<std::string, std::vector<double>>> perPipeline(backends.size());
  for (uint32_t round = 0; round < repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      fused[i].push_back(backends[i]->submitCommand(graphs[i].dispatches()).gpuSeconds * 1000.0);
    }
  for (auto *backend : backends) backend->setDispatchProfiling(true);
  for (uint64_t round = 0; round <= repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      static_cast<void>(backends[i]->submitCommand(graphs[i].dispatches()));
      std::map<std::string, double> run;
      for (const auto &timing : backends[i]->takeDispatchProfile())
        run[timing.pipelineName] += timing.gpuSeconds * 1000.0;
      if (round)
        for (const auto &[name, milliseconds] : run)
          perPipeline[i][name].push_back(milliseconds);
    }
  for (size_t i = 0; i < backends.size(); ++i) {
    backends[i]->setDispatchProfiling(false);
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

std::string json(const Case &item, const std::string &shape, uint32_t history,
                 const std::string &kind, uint32_t lanes, size_t variant) {
  std::string out = "{\"variant\":" + std::to_string(variant) + ",\"shape\":\"" + shape + "\",\"history\":" + std::to_string(history) +
                    ",\"kind\":\"" + kind + "\",\"lanes\":" + std::to_string(lanes) +
                    ",\"fused_ms\":" + std::to_string(item.fusedMilliseconds) +
                    ",\"kv_bytes\":" + std::to_string(item.kvBytes) + ",\"pipelines\":{";
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
                   "[--compare-metallib PATH] [--kv-format int8|bf16]\n";
      return 64;
    }
    std::vector<uint32_t> histories{0, 2048, 8192, 16384, 32768, 65536, 131072};
    std::vector<uint32_t> lanes{1, 4};
    std::vector<std::string> shapes{"27b", "35b"};
    uint32_t repeat = 5;
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
    metal::MetalBackend backend(argv[1]);
    std::unique_ptr<metal::MetalBackend> comparison;
    std::vector<metal::MetalBackend *> backends{&backend};
    if (!comparisonLibrary.empty()) {
      comparison = std::make_unique<metal::MetalBackend>(comparisonLibrary);
      backends.push_back(comparison.get());
    }
    std::cerr << "device " << backend.capabilities().deviceName << ", one attention layer, "
              << "Page32 " << kv::formatName(format) << " KV, median of " << repeat << " fused graphs (ms)\n";
    std::cout << "{\"device\":\"" << backend.capabilities().deviceName << "\",\"kv_format\":\"" << kv::formatName(format) << "\",\"cases\":[";
    bool firstCase = true;
    for (const std::string &shape : shapes) {
      const AttentionShape geometry = shape == "27b" ? AttentionShape{24, 4, 256, format}
                                                     : AttentionShape{16, 2, 256, format};
      const std::string name = shape == "27b" ? "qwen3.8-27b" : "qwen3.6-35b-a3b";
      std::cerr << "\n" << name << "  (" << geometry.queryHeads << " query heads, "
                << geometry.kvHeads << " KV heads, d=" << geometry.headDimension << ")\n";
      for (uint32_t history : histories) {
        auto report = [&](bool prefill, uint32_t lane) {
          const auto cases = measure(backends, makePlan(geometry, prefill, lane, history), repeat);
          for (size_t i = 0; i < cases.size(); ++i) {
            std::cout << (firstCase ? "" : ",")
                      << json(cases[i], name, history, prefill ? "prefill" : "verify", lane, i);
            firstCase = false;
            std::cerr << history << " " << (prefill ? "prefill" : "verify")
                      << " lanes=" << lane << " variant=" << i << " fused="
                      << cases[i].fusedMilliseconds << " ms\n";
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
