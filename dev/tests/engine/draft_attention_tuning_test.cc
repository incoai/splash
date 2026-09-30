#include "tuning/DraftAttentionTuning.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;

constexpr std::array shapes{
    DraftAttentionShape{5120, 1280, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{2048, 512, 6144, 4096, 32, 8, 128}};

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Function> void rejects(Function function) {
  bool rejected = false;
  try { function(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid draft tuning fixture was accepted");
}
uint64_t align(uint64_t bytes) { return (bytes + 16383) / 16384 * 16384; }

uint64_t expectedBytes(DraftAttentionShape shape, uint32_t lanes) {
  const uint64_t rows = uint64_t{lanes} * 8;
  const uint64_t convolution = rows * shape.hiddenSize * 2;
  const uint64_t dynamic = rows * shape.dynamicSize * 2;
  const uint64_t weights = uint64_t{4} * shape.hiddenSize * 2;
  const uint64_t qkv = rows * shape.qkvSize * 2;
  // The grouped tensor carries the attention core's four 32 x 130 fp32
  // split partials per (lane, head) behind the query rows; only the rows
  // are qualified outputs, and the packed reorder holds rows alone.
  const uint64_t rowsBytes = rows * shape.attentionSize * 2;
  const uint64_t grouped = rowsBytes + uint64_t{lanes} * shape.kvHeads * 4 * 16640;
  const uint64_t query = rows * shape.kvHeads * shape.headDimension * 2;
  const uint64_t rope = rows * shape.headDimension / 2 * 4;
  const uint64_t window = uint64_t{lanes} * shape.kvHeads * 2048 * shape.headDimension * 2;
  const uint64_t reference = 2 * (4 * convolution + qkv + 2 * rowsBytes + 2 * query);
  return 8 * align(convolution) + 2 * align(dynamic) + 2 * align(weights) +
         2 * align(qkv) + align(grouped) + align(rowsBytes) + 2 * align(query) +
         2 * align(uint64_t{shape.headDimension} * 2) + 2 * align(rope) +
         2 * align(window) + align(reference);
}

void cpuTests() {
  for (auto shape : shapes) {
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto bytes = draftAttentionTuningFixtureBytes({shape, lanes});
      require(bytes == expectedBytes(shape, lanes) && bytes % 16384 == 0,
              "draft fixture admission differs from exact tensor/reference bound");
      require(bytes < 64ULL * 1024 * 1024,
              "draft tuning fixture unexpectedly duplicated its window histories");
    }
    rejects([&] { (void)draftAttentionTuningFixtureBytes({shape, 0}); });
    rejects([&] { (void)draftAttentionTuningFixtureBytes({shape, 5}); });
    rejects([&] { (void)draftAttentionTuningFixtureBytes({shape, UINT32_MAX}); });
  }
  rejects([] { (void)draftAttentionTuningFixtureBytes({{}, 1}); });
  auto invalid = shapes[0];
  invalid.dynamicSize = 512;
  rejects([&] { (void)draftAttentionTuningFixtureBytes({invalid, 1}); });
  invalid = shapes[0];
  invalid.headDimension = 256;
  rejects([&] { (void)draftAttentionTuningFixtureBytes({invalid, 1}); });
}

void checkResult(const DraftAttentionTuningResult &result,
                 const MeasurementOptions &options) {
  if (result.failure) std::rethrow_exception(result.failure);
  // Every candidate but the full-grid baseline is measured on both history
  // workloads, in candidate-major order.
  const auto candidates = DraftAttention::candidates(result.choice.workload.shape);
  const size_t alternatives = candidates.size() - 1;
  constexpr size_t histories = 2;
  require(result.complete && result.measurements.size() == alternatives * histories,
          "draft tuner did not finish every candidate on both history workloads");
  std::vector<WorkloadMeasurements> gpu(result.measurements.size()),
      wall(result.measurements.size());
  for (size_t i = 0; i < result.measurements.size(); ++i) {
    const auto &m = result.measurements[i];
    require((m.status == MeasurementStatus::Completed || m.status == MeasurementStatus::Rejected) &&
                m.candidate.value == i / histories + 1 && m.workload.value == i % histories &&
                m.pairCount == kMinPairedSamples && m.warmup.returnedCalls == 2 &&
                m.measurement.returnedCalls == 24,
            "draft tuner omitted raw complete-stage timing pairs");
    for (size_t pair = 0; pair < m.pairCount; ++pair)
      require(m.gpuPairs[pair].first == measurementOrder(pair) &&
                  m.wallPairs[pair].first == measurementOrder(pair),
              "draft tuner changed paired measurement order");
    gpu[i] = {m.workload, m.rawGpuSamples()};
    wall[i] = {m.workload, m.rawWallSamples()};
  }
  std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
  for (size_t candidate = 1; candidate < candidates.size(); ++candidate) {
    const size_t offset = (candidate - 1) * histories;
    gpuCandidates.push_back({{static_cast<uint32_t>(candidate)}, {gpu.data() + offset, histories}});
    wallCandidates.push_back({{static_cast<uint32_t>(candidate)}, {wall.data() + offset, histories}});
  }
  constexpr std::array required{WorkloadId{0}, WorkloadId{1}};
  const auto gpuWinner = selectCandidate(gpuCandidates, required, options.policy);
  const auto wallWinner = selectCandidate(wallCandidates, required, options.policy);
  DraftAttentionConfiguration expected;
  if (gpuWinner.verdict == SelectionVerdict::Selected &&
      wallWinner.verdict == SelectionVerdict::Selected &&
      gpuWinner.candidate == wallWinner.candidate)
    expected = candidates[gpuWinner.candidate.value];
  require(result.choice.configuration == expected,
          "draft selected without independent two-history GPU/wall agreement");
}

// Explicit opt-in. Timing rejection is allowed; this asserts completeness,
// output qualification, accounting and selection policy, not performance.
void metalTests(const char *metallib) {
  metal::MetalBackend backend(metallib);
  MeasurementOptions options;
  options.warmupPairs = 1;
  options.maximumWallSeconds = 60;
  size_t calls = 0;
  uint64_t admittedBytes = 0;
  const metal::AllocationAdmission admit = [&](uint64_t bytes, const auto &allocate) {
    ++calls;
    admittedBytes = bytes;
    allocate();
    return true;
  };
  for (auto shape : shapes)
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const DraftAttentionWorkload workload{shape, lanes};
      const size_t before = calls;
      const auto result = tuneDraftAttention(backend, admit, workload, options);
      checkResult(result, options);
      require(calls == before + 1 &&
                  admittedBytes == draftAttentionTuningFixtureBytes(workload),
              "draft fixtures were not allocated once under exact admission");
    }
  const DraftAttentionWorkload workload{shapes[0], 3};
  const auto denied = tuneDraftAttention(backend,
      [](uint64_t, const auto &) { return false; }, workload, options);
  require(!denied.complete && denied.measurements.empty() && !denied.failure &&
              denied.choice.configuration == DraftAttentionConfiguration{},
          "draft admission denial did not retain baseline without commands");
  const auto invalidAdmission = tuneDraftAttention(backend,
      [](uint64_t, const auto &) { return true; }, workload, options);
  require(!invalidAdmission.complete && invalidAdmission.failure &&
              invalidAdmission.measurements.empty(),
          "draft accepted successful admission without an allocation");
  const size_t before = calls;
  const auto stopped = tuneDraftAttention(backend, admit, workload, options, {}, [] { return true; });
  const auto pressured = tuneDraftAttention(backend, admit, workload, options, [] { return true; });
  require(!stopped.complete && !pressured.complete && calls == before,
          "draft stop/pressure allocated or submitted work");
  auto invalid = options;
  invalid.samplePairs = 1;
  const auto rejected = tuneDraftAttention(backend, admit, workload, invalid);
  require(!rejected.complete && rejected.measurements.empty() && calls == before,
          "invalid draft measurement options consumed fixture resources");
  const auto cancelled = tuneDraftAttention(backend, admit, workload, options, {},
      [&] { return calls != before; });
  require(!cancelled.complete && cancelled.measurements.empty(),
          "draft cancellation after admission submitted commands");
}
} // namespace

int main(int argc, char **argv) {
  try {
    cpuTests();
    if (argc == 3 && std::string_view(argv[1]) == "--metal") metalTests(argv[2]);
    else require(argc == 1, "usage: draft-attention-tuning [--metal production.metallib]");
    std::cout << "PASS draft attention tuning: " <<
        (argc == 1 ? "CPU fixture bounds" : "CPU + native group qualification") << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL draft attention tuning: " << error.what() << '\n';
    return 1;
  }
}
