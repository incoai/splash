// Root-run GPU oracle. --help and --cpu-self-test create no Metal backend.
#include "flash/FlashBatchPrefill.hpp"
#include "flash/FlashAffineMPP.hpp"
#include "flash/FlashMoE.hpp"
#include "engine/Json.hpp"
#include "metal/ProfilingJson.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash;
using namespace splash::flash;
constexpr uint32_t kHyper = 10240, kVocabulary = 248320;
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
float f32(uint16_t value) { return std::bit_cast<float>(uint32_t{value} << 16); }
template <class T> std::vector<T> copied(const metal::MetalBuffer &buffer, uint64_t bytes) {
  require(buffer && buffer.contents() && bytes % sizeof(T) == 0 && buffer.sizeBytes() >= bytes,
      "oracle received an invalid borrowed output");
  const auto *base = static_cast<const T *>(buffer.contents());
  return {base, base + bytes / sizeof(T)};
}
uint32_t greedy(std::span<const uint16_t> logits) {
  require(!logits.empty(), "greedy received no logits");
  uint32_t best = 0;
  for (uint32_t index = 1; index < logits.size(); ++index)
    if (f32(logits[index]) > f32(logits[best])) best = index;
  return best;
}
std::vector<uint32_t> selected(const char *name, std::vector<uint32_t> fallback, uint32_t maximum) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  std::stringstream input(raw); std::vector<uint32_t> result; std::string item;
  while (std::getline(input, item, ',')) {
    size_t used = 0;
    const auto value = std::stoul(item, &used);
    require(used == item.size() && value >= 1 && value <= maximum,
        std::string(name) + " contains an invalid case selector");
    result.push_back(static_cast<uint32_t>(value));
  }
  require(!result.empty() && std::string_view(raw).back() != ',', std::string(name) + " is empty");
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}
double tolerance(const char *name, double fallback) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  size_t used = 0; const double value = std::stod(raw, &used);
  require(used == std::strlen(raw) && std::isfinite(value) && value > 0 && value <= 0.1,
      std::string(name) + " must be a finite value in (0,0.1]");
  return value;
}
bool flag(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  require(std::string_view(raw) == "0" || std::string_view(raw) == "1", std::string(name) + " must be 0 or 1");
  return std::string_view(raw) == "1";
}
std::string profileMode() {
  const char *raw = std::getenv("FLASH_BATCH_PREFILL_ORACLE_PROFILE");
  const std::string mode = raw ? raw : "off";
  require(mode == "off" || mode == "command" || mode == "dispatch" || mode == "stage",
      "FLASH_BATCH_PREFILL_ORACLE_PROFILE must be off, command, dispatch or stage");
  return mode;
}
std::string family(std::string_view name) {
  if (name.starts_with("flash_gdn")) return "gdn";
  if (name.starts_with("flash_qsa")) return "qsa";
  if (name.starts_with("flash_moe") || name.starts_with("flash_expert")) return "moe";
  if (name.starts_with("flash_ple")) return "ple";
  if (name.starts_with("flash_hc") || name.starts_with("flash_forward_hc")) return "hc";
  if (name == "flash_affine_embedding") return "embedding";
  if (name == "flash_forward_copy_words") return "copies";
  if (name.starts_with("flash_affine") || name.starts_with("flash_dense") || name.starts_with("flash_float_dense"))
    return "dense";
  return "other";
}
void writeProfile(std::ostream &out, const metal::CommandDispatchProfile &command,
    std::string_view phase, std::string_view label, uint32_t lanes, uint32_t rows) {
  struct Attribution { uint64_t dispatches = 0, timed = 0; double seconds = 0; };
  std::map<std::string, Attribution> families;
  for (const auto &dispatch : command.dispatches) {
    auto &entry = families[family(dispatch.pipelineName)];
    ++entry.dispatches;
    if (dispatch.timestampsValid) { ++entry.timed; entry.seconds += dispatch.gpuSeconds; }
  }
  out << "{\"phase\":" << json::quote(phase) << ",\"case\":" << json::quote(label)
      << ",\"lanes\":" << lanes << ",\"real_rows_per_lane\":" << rows
      << ",\"attribution_scope\":\"descriptive sampled dispatches; stage/dispatch modes perturb encoding; not a throughput baseline\""
      << ",\"families\":{";
  bool comma = false;
  for (const auto &[name, entry] : families) {
    if (comma) out << ','; comma = true;
    out << json::quote(name) << ":{\"dispatches\":" << entry.dispatches << ",\"timed_dispatches\":"
        << entry.timed << ",\"gpu_seconds\":"; profiling::writeNumber(out, entry.seconds); out << '}';
  }
  out << "},\"command\":"; profiling::writeJson(out, command); out << "}\n";
}
struct Comparison final {
  uint64_t comparisons = 0, elements = 0, differences = 0, statePlanes = 0,
      tailBytes = 0, greedyComparisons = 0, greedyMatches = 0;
  double maxL2 = 0, minimumCosine = 1, limit = 0.015, cosineLimit = 0.9999;
  bool exact = false, pass = true;
  std::string worst;
  template <class T, class Decode> void compare(std::span<const T> actual,
      std::span<const T> expected, Decode decode, const std::string &name) {
    require(actual.size() == expected.size(), "comparison shape differs: " + name);
    double error = 0, expectedNorm = 0, actualNorm = 0, dot = 0;
    uint64_t different = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
      const double a = decode(actual[index]), e = decode(expected[index]);
      if (!std::isfinite(a) || !std::isfinite(e))
        throw std::runtime_error("nonfinite comparison value: " + name);
      different += actual[index] != expected[index];
      error += (a - e) * (a - e); expectedNorm += e * e; actualNorm += a * a; dot += a * e;
    }
    const double relative = std::sqrt(error / std::max(1e-30, expectedNorm));
    const double cosine = actualNorm == 0 && expectedNorm == 0 ? 1.0
        : dot / std::max(1e-30, std::sqrt(actualNorm * expectedNorm));
    ++comparisons; elements += actual.size(); differences += different;
    if (relative >= maxL2) { maxL2 = relative; worst = name; }
    minimumCosine = std::min(minimumCosine, cosine);
    pass &= relative <= limit && cosine >= cosineLimit && (!exact || different == 0);
  }
  void bf(std::span<const uint16_t> actual, std::span<const uint16_t> expected, const std::string &name) {
    compare(actual, expected, [](uint16_t value) { return double(f32(value)); }, name);
  }
  void float32(std::span<const float> actual, std::span<const float> expected, const std::string &name) {
    compare(actual, expected, [](float value) { return double(value); }, name);
  }
  void logits(std::span<const uint16_t> actual, std::span<const uint16_t> expected, const std::string &name) {
    bf(actual, expected, name); ++greedyComparisons; greedyMatches += greedy(actual) == greedy(expected);
  }
  void write(std::ostream &out) const {
    out << "{\"pass\":" << (pass ? "true" : "false")
        << ",\"exact\":" << (exact ? "true" : "false")
        << ",\"relative_l2_limit\":" << limit << ",\"minimum_cosine_limit\":" << cosineLimit
        << ",\"comparisons\":" << comparisons << ",\"elements\":" << elements
        << ",\"bit_differences\":" << differences << ",\"maximum_relative_l2\":" << maxL2
        << ",\"minimum_cosine\":" << minimumCosine << ",\"worst_plane\":" << json::quote(worst)
        << ",\"state_planes\":" << statePlanes << ",\"untouched_tail_bytes\":" << tailBytes
        << ",\"greedy_comparisons\":" << greedyComparisons << ",\"greedy_matches\":" << greedyMatches << '}';
  }
};
std::vector<uint32_t> loadTokens(const char *path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
  require(data != nil, "could not read prompt JSON");
  NSError *error = nil;
  id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [object isKindOfClass:[NSArray class]], "prompt must be a JSON array");
  std::vector<uint32_t> result;
  for (id entry in static_cast<NSArray *>(object)) {
    require([entry isKindOfClass:[NSNumber class]] &&
        CFGetTypeID((__bridge CFTypeRef)entry) != CFBooleanGetTypeID(), "prompt token must be an integer");
    const double value = static_cast<NSNumber *>(entry).doubleValue;
    require(std::isfinite(value) && value >= 0 && value < kVocabulary && std::floor(value) == value,
        "invalid prompt token");
    result.push_back(static_cast<uint32_t>(value));
  }
  require(!result.empty(), "prompt JSON has no tokens");
  return result;
}
template <class Operation> void invalid(metal::MetalBackend &backend, Operation operation) {
  const uint64_t submissions = backend.submissionCount();
  bool rejected = false;
  try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected && submissions == backend.submissionCount(), "invalid call was not rejected before GPU submission");
}
// Check every byte without constructing diagnostics in the success path.
// memcpy permits unaligned views and avoids aliasing BF16/F32 storage as U64.
// A nonzero block is inspected bytewise to preserve the exact failure offset.
size_t firstNonZeroByte(std::span<const uint8_t> bytes) {
  size_t offset = 0;
  while (bytes.size() - offset >= 4 * sizeof(uint64_t)) {
    std::array<uint64_t, 4> words;
    std::memcpy(words.data(), bytes.data() + offset, sizeof(words));
    if (words[0] | words[1] | words[2] | words[3]) {
      for (size_t byte = 0; byte < sizeof(words); ++byte)
        if (bytes[offset + byte]) return offset + byte;
    }
    offset += sizeof(words);
  }
  for (; offset < bytes.size(); ++offset)
    if (bytes[offset]) return offset;
  return bytes.size();
}
void compareState(FlashBatchPrefill &batch, const FlashRequestState &actual,
    const FlashRequestState &expected, Comparison &comparison, const std::string &label) {
  const auto a = batch.inspectState(actual), e = batch.inspectState(expected);
  require(a.size() == e.size(), "state inspection sizes differ");
  for (size_t index = 0; index < a.size(); ++index) {
    require(a[index].name == e[index].name && a[index].dtype == e[index].dtype &&
        a[index].activeBytes == e[index].activeBytes, "state inspection layouts differ");
    const auto &plane = a[index]; const uint64_t bytes = plane.activeBytes;
    ++comparison.statePlanes;
    const auto name = label + "." + plane.name;
    if (bytes && plane.dtype == FlashDType::BF16)
      comparison.bf(copied<uint16_t>(plane.buffer, bytes), copied<uint16_t>(e[index].buffer, bytes), name);
    else if (bytes && plane.dtype == FlashDType::F32)
      comparison.float32(copied<float>(plane.buffer, bytes), copied<float>(e[index].buffer, bytes), name);
    else if (bytes) {
      require(std::memcmp(plane.buffer.contents(), e[index].buffer.contents(), bytes) == 0,
          "integer state plane differs: " + name);
    }
    for (const auto &current : {plane, e[index]}) {
      require(current.activeBytes <= current.buffer.sizeBytes(), "state live extent exceeds its allocation");
      const auto *base = static_cast<const uint8_t *>(current.buffer.contents());
      const uint64_t tailBytes = current.buffer.sizeBytes() - current.activeBytes;
      const size_t changed = firstNonZeroByte({base + current.activeBytes, size_t(tailBytes)});
      if (changed != tailBytes)
        throw std::runtime_error("request capacity tail was modified: " + name +
            " byte " + std::to_string(current.activeBytes + changed));
      comparison.tailBytes += tailBytes;
    }
  }
}
void cpuSelfTest() {
  require(FlashBatchPrefill::workspacePlannedBytes(8192, 4, 512) >
      FlashBatchPrefill::workspacePlannedBytes(8192, 1, 512), "workspace plan does not scale with real lanes");
  require(FlashBatchPrefill::workspacePlannedBytes(8192, 4, 2048) >
          FlashBatchPrefill::workspacePlannedBytes(8192, 4, 1024) &&
          FlashBatchPrefill::workspacePlannedBytes(8192, 4, 1024) >
          FlashBatchPrefill::workspacePlannedBytes(8192, 4, 512) &&
          uint64_t{kFlashBatchPrefillMaximumPhysicalRows} * 10 * 2560 * 2 == 419430400 &&
          kFlashBatchPrefillMaximumPhysicalRows == 4 * kFlashBatchPrefillMaximumRowsPerLane,
      "wide prefill arena must use checked 8192-row/81920-route extents");
  const auto invalidMessage = [&](auto operation) {
    try { operation(); } catch (const std::invalid_argument &error) { return std::string(error.what()); }
    throw std::runtime_error("CPU geometry probe unexpectedly accepted empty buffers");
  };
  metal::CommandGraph graph;
  require(invalidMessage([&] { addRoute(graph, {}, {}, {}, {}, 8192, 512, 10); })
              == "Flash MoE insufficient router logits" &&
          invalidMessage([&] { addRoute(graph, {}, {}, {}, {}, 8193, 512, 10); })
              == "Flash MoE invalid routing geometry",
      "routing must accept physical row8192 geometry and reject row8193 before buffer work");
  FlashAffineProjection projection;
  projection.experts = 1; projection.outputSize = 64; projection.inputSize = 64;
  projection.bits = 4; projection.groupSize = 32;
  projection.weightRowStrideBytes = 32; projection.parameterRowStrideBytes = 4;
  require(invalidMessage([&] { addAffineMPP(graph, {}, projection, {}, {}, 8192,
              FlashAffineMPPMode::ReconstructedBF16, FlashAffineMPPTile::M32N64); })
              == "Flash MPP affine invalid weights" &&
          invalidMessage([&] { addAffineMPP(graph, {}, projection, {}, {}, 8193,
              FlashAffineMPPMode::ReconstructedBF16, FlashAffineMPPTile::M32N64); })
              == "Flash MPP affine invalid dense geometry/mode",
      "matrix geometry must accept physical row8192 and reject row8193 before operand work");
  for (const auto shape : std::array<std::array<uint32_t, 3>, 6>{{{0, 4, 512}, {262145, 4, 512},
      {8192, 0, 512}, {8192, 5, 512}, {8192, 4, 0}, {8192, 4, 2049}}}) {
    bool rejected = false;
    try { (void)FlashBatchPrefill::workspacePlannedBytes(shape[0], shape[1], shape[2]); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid admission geometry was accepted");
  }
  Comparison good; const std::array<uint16_t, 3> a{0x3f80, 0x4000, 0xc000};
  good.logits(a, a, "equal"); require(good.pass && good.maxL2 == 0 && good.greedyMatches == 1, "equal comparator failed");
  Comparison bad; const std::array<uint16_t, 3> b{0x4080, 0x4000, 0xc000};
  bad.logits(b, a, "different"); require(!bad.pass && bad.greedyMatches == 0, "different comparator failed");
  require(family("flash_gdn_staged") == "gdn" && family("flash_qsa_online") == "qsa" &&
      family("flash_moe_q4x8_gate_up_m32_n64") == "moe" && family("flash_dense_cache_m32_n128") == "dense",
      "trace family attribution is invalid");
  std::array<uint8_t, 131> guards{};
  for (size_t begin = 0; begin < 8; ++begin) for (size_t count = 0; count <= 123; ++count) {
    const auto view = std::span<const uint8_t>(guards).subspan(begin, count);
    require(firstNonZeroByte(view) == count, "zero-tail scanner rejected zero bytes");
    for (size_t byte = 0; byte < count; ++byte) {
      guards[begin + byte] = 0x80;
      require(firstNonZeroByte(view) == byte, "zero-tail scanner missed a changed byte");
      guards[begin + byte] = 0;
    }
  }
  std::cout << "Flash batch prefill CPU self-test passed\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::string reportPath, routes, fingerprint, error, requestedProfile = "off";
    uint32_t windows = 1;
    bool gpuFeatureCopy = false;
    Comparison comparison;
    uint64_t scenarios = 0, batchCommands = 0, scalarCommands = 0, rejections = 0;
    double batchGpu = 0, scalarGpu = 0, batchWall = 0, scalarWall = 0;
    std::vector<std::string> records;
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "usage: flash-batch-prefill-oracle METALLIB PACKAGE TOKEN_JSON REPORT_JSON\n"
          "Real uniform rows 1..2048, lanes 1..4; default lanes4/rows512.\n"
          "Select FLASH_BATCH_PREFILL_ORACLE_LANES and _ROWS (comma-separated).\n"
          "_WINDOWS=1..4 repeats real windows on the same lane states.\n"
          "_OFFSETS=0 skips the independent-offset case (default1).\n"
          "_LANE_TOKEN_PREFIX reads PREFIX0.json..PREFIX3.json as exact lane fixtures.\n"
          "_PROFILE=command|dispatch|stage writes REPORT_JSON.trace.jsonl.\n"
          "_GPU_FEATURE_COPY=1 checks owned GPU destination delivery and rejection guards.\n"
          "Each case compares cold and independent lane-offset prefill, all\n"
          "premixer rows, last logits, F32 recurrent/BF16 cache states, zero\n"
          "capacity tails, guards, then three fixed-token AR continuations.\n"
          "Uses current route flags without silently changing arithmetic.\n"
          "_EXACT=1 requires identical bits; _MAX_L2 defaults0.015.\n"
          "GPU command timings include startup pipeline warmup; these are\n"
          "numerical qualification timings, not a balanced performance claim.\n";
        return 0;
      }
      require(argc == 5, "usage: flash-batch-prefill-oracle METALLIB PACKAGE TOKEN_JSON REPORT_JSON");
      reportPath = argv[4];
      requestedProfile = profileMode();
      const auto windowChoice = selected("FLASH_BATCH_PREFILL_ORACLE_WINDOWS", {1}, 4);
      require(windowChoice.size() == 1, "FLASH_BATCH_PREFILL_ORACLE_WINDOWS requires one integer 1..4");
      windows = windowChoice.front();
      const bool independentOffsets = flag("FLASH_BATCH_PREFILL_ORACLE_OFFSETS", true);
      gpuFeatureCopy = flag("FLASH_BATCH_PREFILL_ORACLE_GPU_FEATURE_COPY", false);
      const auto lanesList = selected("FLASH_BATCH_PREFILL_ORACLE_LANES", {4}, 4);
      const auto rowsList = selected("FLASH_BATCH_PREFILL_ORACLE_ROWS", {512}, kFlashBatchPrefillMaximumRowsPerLane);
      const uint32_t maximumRows = *std::max_element(rowsList.begin(), rowsList.end());
      require(uint64_t{windows} * maximumRows + 32 <= 8192,
          "oracle windows plus offset/continuation budget exceed context");
      comparison.exact = flag("FLASH_BATCH_PREFILL_ORACLE_EXACT", false);
      comparison.limit = tolerance("FLASH_BATCH_PREFILL_ORACLE_MAX_L2", 0.015);
      const auto fixture = loadTokens(argv[3]);
      std::array<std::vector<uint32_t>, 4> laneFixtures;
      if (const char *prefix = std::getenv("FLASH_BATCH_PREFILL_ORACLE_LANE_TOKEN_PREFIX"))
        for (uint32_t lane = 0; lane < 4; ++lane)
          laneFixtures[lane] = loadTokens((std::string(prefix) + std::to_string(lane) + ".json").c_str());
      metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      FlashForward trunk(backend, weights, 8192, maximumRows, 4);
      FlashBatchPrefill batch(backend, weights, trunk, 8192, 4, maximumRows);
      routes = batch.kernelRoutes(); fingerprint = weights.manifestFingerprint();
      require(batch.workspaceBytes() <= FlashBatchPrefill::workspacePlannedBytes(8192, 4, maximumRows),
          "batch prefill allocated bytes exceed admission estimate");
      std::ofstream trace;
      if (requestedProfile != "off") {
        const auto mode = requestedProfile == "stage" ? metal::CommandDispatchProfilingMode::StagePerDispatch
            : requestedProfile == "dispatch" ? metal::CommandDispatchProfilingMode::DispatchBoundary
            : metal::CommandDispatchProfilingMode::Command;
        require(backend.commandDispatchProfilingCapability().supports(mode), "requested profiling mode is unsupported");
        backend.setCommandDispatchProfiling(mode);
        trace.open(reportPath + ".trace.jsonl");
        require(bool(trace), "could not open batch prefill trace");
      }
      const auto takeProfiles = [&](const char *phase, const std::string &label, uint32_t lanes, uint32_t rows) {
        if (!trace.is_open()) return;
        const auto profiles = backend.takeCommandDispatchProfiles();
        require(profiles.size() == 1, "missing or dropped batch prefill command profile");
        writeProfile(trace, profiles.front(), phase, label, lanes, rows);
        trace.flush(); require(bool(trace), "could not write batch prefill trace");
      };
      const auto tokenAt = [&](uint32_t lane, uint64_t position) {
        const auto &source = laneFixtures[lane].empty() ? fixture : laneFixtures[lane];
        return source[(position + (laneFixtures[lane].empty() ? lane * 71 : 0)) % source.size()];
      };
      metal::MetalBuffer lastBatchArenaLogits;
      for (uint32_t lanes : lanesList) for (uint32_t rows : rowsList) for (bool offsets : {false, true}) {
        if (offsets && !independentOffsets) continue;
        std::array<FlashRequestState, 4> actual, reference;
        std::vector<FlashRequestState *> pointers;
        std::vector<uint32_t> tokens;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          actual[lane] = trunk.createState(); reference[lane] = trunk.createState(); pointers.push_back(&actual[lane]);
          if (offsets) {
            std::vector<uint32_t> prior;
            for (uint32_t row = 0; row < 7 + lane * 3; ++row) prior.push_back(tokenAt(lane, row));
            auto a = trunk.forward(actual[lane], prior);
            takeProfiles("offset_prefill", "actual.lane" + std::to_string(lane), 1, prior.size());
            auto e = trunk.forward(reference[lane], prior);
            takeProfiles("offset_prefill", "reference.lane" + std::to_string(lane), 1, prior.size());
            require(a.logicalLength == e.logicalLength, "offset fixture lengths differ");
          }
        }
        for (uint32_t window = 0; window < windows; ++window) {
        tokens.clear();
        for (uint32_t lane = 0; lane < lanes; ++lane)
          for (uint32_t row = 0; row < rows; ++row) tokens.push_back(tokenAt(lane, actual[lane].logicalLength() + row));
        const std::string label = "lanes" + std::to_string(lanes) + ".rows" + std::to_string(rows) +
            (offsets ? ".offset" : ".cold") + ".window" + std::to_string(window);
        constexpr uint64_t copyGuardBytes = 64;
        const uint64_t hiddenBytes = flashBatchPrefillHiddenCopyBytes(lanes, rows);
        metal::MetalBuffer destinationAllocation, destination;
        if (gpuFeatureCopy) {
          destinationAllocation = backend.allocateBuffer(hiddenBytes + 2 * copyGuardBytes,
              metal::BufferStorage::Shared, "oracle owned GPU prefill features and guards");
          std::memset(destinationAllocation.contents(), 0xa7, destinationAllocation.sizeBytes());
          // Deliberately expose a longer capacity view: only real rows may be
          // returned/written, and the caller's trailing guard must survive.
          destination = backend.view(destinationAllocation, copyGuardBytes, hiddenBytes + copyGuardBytes);
        }
        const auto result = batch.forwardBatch(pointers, tokens, rows, true, destination);
        lastBatchArenaLogits = result.logitsBF16;
        require(result.hiddenDeliveredToDestination == gpuFeatureCopy &&
                (!gpuFeatureCopy || result.hiddenBF16.sameView(backend.view(destination, 0, hiddenBytes))),
            "batch feature destination delivery metadata differs");
        destination = {}; destinationAllocation = {};
        const auto requireCopyGuards = [&] {
          if (!gpuFeatureCopy) return;
          const auto *begin = static_cast<const uint8_t *>(result.hiddenBF16.contents());
          const auto *prefix = begin - copyGuardBytes;
          for (uint64_t byte = 0; byte < copyGuardBytes; ++byte)
            require(prefix[byte] == 0xa7 && begin[hiddenBytes + byte] == 0xa7,
                "owned GPU feature copy changed an exterior guard");
        };
        requireCopyGuards();
        takeProfiles("batch_prefill", label, lanes, rows);
        ++scenarios; ++batchCommands; batchGpu += result.timing.gpuSeconds; batchWall += result.timing.wallSeconds;
        require(result.lanes == lanes && result.rows == rows && result.capacity == 8192 && result.logicalLengths.size() == lanes,
            "batch result real-row metadata differs");
        const auto actualHidden = copied<uint16_t>(result.hiddenBF16, uint64_t{lanes} * rows * kHyper * 2);
        const auto actualLogits = copied<uint16_t>(result.logitsBF16, uint64_t{lanes} * kVocabulary * 2);
        std::ostringstream record;
        record << "{\"case\":" << json::quote(label) << ",\"lanes\":" << lanes << ",\"real_rows_per_lane\":" << rows
            << ",\"batch_gpu_seconds\":" << result.timing.gpuSeconds << ",\"batch_wall_seconds\":" << result.timing.wallSeconds;
        double expectedGpu = 0, expectedWall = 0;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const auto expected = trunk.forward(reference[lane], std::span<const uint32_t>(tokens).subspan(lane * rows, rows), false, true);
          takeProfiles("sequential_prefill", label + ".lane" + std::to_string(lane), 1, rows);
          ++scalarCommands; expectedGpu += expected.timing.gpuSeconds; expectedWall += expected.timing.wallSeconds;
          comparison.bf(std::span<const uint16_t>(actualHidden).subspan(uint64_t{lane} * rows * kHyper, uint64_t{rows} * kHyper),
              copied<uint16_t>(expected.hiddenBF16, uint64_t{rows} * kHyper * 2), label + ".hidden.lane" + std::to_string(lane));
          comparison.logits(std::span<const uint16_t>(actualLogits).subspan(uint64_t{lane} * kVocabulary, kVocabulary),
              copied<uint16_t>(expected.logitsBF16, uint64_t{kVocabulary} * 2), label + ".logits.lane" + std::to_string(lane));
          require(actual[lane].logicalLength() == reference[lane].logicalLength() &&
              result.logicalLengths[lane] == actual[lane].logicalLength() && trunk.ownsState(actual[lane]), "prefill state metadata differs");
          compareState(batch, actual[lane], reference[lane], comparison, label + ".lane" + std::to_string(lane));
          for (uint32_t step = 0; window + 1 == windows && step < 3; ++step) {
            const std::array<uint32_t, 1> continuation{tokenAt(lane, reference[lane].logicalLength())};
            const auto a = trunk.forward(actual[lane], continuation, false, true);
            takeProfiles("continuation", label + ".actual.lane" + std::to_string(lane), 1, 1);
            const auto hidden = copied<uint16_t>(a.hiddenBF16, uint64_t{kHyper} * 2);
            const auto logits = copied<uint16_t>(a.logitsBF16, uint64_t{kVocabulary} * 2);
            const auto e = trunk.forward(reference[lane], continuation, false, true);
            takeProfiles("continuation", label + ".reference.lane" + std::to_string(lane), 1, 1);
            comparison.bf(hidden, copied<uint16_t>(e.hiddenBF16, uint64_t{kHyper} * 2), label + ".continuation.hidden");
            comparison.logits(logits, copied<uint16_t>(e.logitsBF16, uint64_t{kVocabulary} * 2), label + ".continuation.logits");
            require(a.logicalLength == e.logicalLength && !actual[lane].poisoned(), "continuation state metadata differs");
          }
          if (window + 1 == windows)
            compareState(batch, actual[lane], reference[lane], comparison, label + ".continued.lane" + std::to_string(lane));
        }
        if (gpuFeatureCopy) {
          requireCopyGuards();
          require(std::memcmp(result.hiddenBF16.contents(), actualHidden.data(), hiddenBytes) == 0,
              "retained owned GPU features changed after later source calls");
        }
        scalarGpu += expectedGpu; scalarWall += expectedWall;
        require(batch.canariesIntact(), "batch prefill scratch guards were modified");
        record << ",\"sequential_gpu_seconds\":" << expectedGpu << ",\"sequential_wall_seconds\":" << expectedWall << '}';
        records.push_back(record.str());
          }
        }
      // Invalid requests must be rejected without submitting or changing a
      // healthy sibling. Constructor context mismatch needs no duplicate cache.
      auto a = trunk.createState(), b = trunk.createState();
      std::array<FlashRequestState *, 2> pair{&a, &b}, duplicate{&a, &a}, null{&a, nullptr};
      const std::array<uint32_t, 2> input{fixture.front(), fixture.back()};
      const auto reject = [&](auto operation) { invalid(backend, operation); ++rejections;
        require(a.logicalLength() == 0 && b.logicalLength() == 0 && !a.poisoned() && !b.poisoned(), "invalid call mutated a state"); };
      reject([&] { (void)batch.forwardBatch({}, {}, 1); });
      reject([&] { (void)batch.forwardBatch(pair, input, 0); });
      reject([&] { (void)batch.forwardBatch(pair, input, maximumRows + 1); });
      reject([&] { (void)batch.forwardBatch(pair, std::span<const uint32_t>(input).first(1), 1); });
      reject([&] { (void)batch.forwardBatch(duplicate, input, 1); });
      reject([&] { (void)batch.forwardBatch(null, input, 1); });
      auto invalidToken = input; invalidToken[1] = kVocabulary;
      reject([&] { (void)batch.forwardBatch(pair, invalidToken, 1); });
      if (gpuFeatureCopy) {
        const uint64_t bytes = flashBatchPrefillHiddenCopyBytes(2, 1);
        auto storage = backend.allocateBuffer(bytes + 8, metal::BufferStorage::Shared,
            "oracle invalid feature destinations");
        reject([&] { (void)batch.forwardBatch(pair, input, 1, false, storage); });
        reject([&] { (void)batch.forwardBatch(pair, input, 1, true, backend.view(storage, 0, bytes - 4)); });
        reject([&] { (void)batch.forwardBatch(pair, input, 1, true, backend.view(storage, 2, bytes)); });
        reject([&] { (void)batch.forwardBatch(pair, input, 1, true, lastBatchArenaLogits); });
        reject([&] { (void)batch.forwardBatch(pair, input, 1, true,
            weights.projection("language_model.model.embed_tokens").weights->buffer); });
        if (const auto *vocabulary = trunk.cachedVocabulary())
          reject([&] { (void)batch.forwardBatch(pair, input, 1, true, vocabulary->buffer); });
        if (const auto *vocabulary = trunk.cachedFloatVocabulary())
          reject([&] { (void)batch.forwardBatch(pair, input, 1, true, vocabulary->buffer); });
        for (const auto &plane : batch.inspectState(a))
          if (plane.buffer.sizeBytes() >= bytes)
            reject([&] { (void)batch.forwardBatch(pair, input, 1, true, plane.buffer); });
        auto privateStorage = backend.allocateBuffer(bytes, metal::BufferStorage::Private,
            "oracle rejected Private feature destination");
        reject([&] { (void)batch.forwardBatch(pair, input, 1, true, privateStorage); });
        metal::MetalBackend otherBackend(argv[1]);
        auto foreignStorage = otherBackend.allocateBuffer(bytes, metal::BufferStorage::Shared,
            "oracle rejected foreign feature destination");
        reject([&] { (void)batch.forwardBatch(pair, input, 1, true, foreignStorage); });
      }
      FlashRequestState foreign;
      std::array<FlashRequestState *, 2> foreignPair{&a, &foreign};
      reject([&] { (void)batch.forwardBatch(foreignPair, input, 1); });
      auto pending = trunk.createState();
      const auto trial = trunk.verify(pending, std::span<const uint32_t>(input).first(1)); (void)trial;
      takeProfiles("safety_verify", "pending", 1, 1);
      std::array<FlashRequestState *, 2> pendingPair{&a, &pending};
      reject([&] { (void)batch.forwardBatch(pendingPair, input, 1); }); trunk.abortVerify(pending);
      if (gpuFeatureCopy)
        reject([&] { (void)batch.forwardBatch(pair, input, 1, true, trial.logitsBF16); });
      reject([&] { FlashBatchPrefill wrong(backend, weights, trunk, 8191); });
      require(batch.canariesIntact() && backend.healthy(), "oracle ended with damaged guards or unhealthy backend");
    } catch (const std::exception &failure) { error = failure.what(); comparison.pass = false; }
    if (!reportPath.empty()) {
      std::ofstream output(reportPath);
      output << "{\"schema\":\"splash-flash-batch-prefill-oracle-v1\",\"pass\":" << (comparison.pass && error.empty() ? "true" : "false")
          << ",\"error\":" << json::quote(error) << ",\"manifest_fingerprint\":" << json::quote(fingerprint)
          << ",\"kernel_routes\":" << json::quote(routes) << ",\"scenarios\":" << scenarios
          << ",\"profile_mode\":" << json::quote(requestedProfile) << ",\"windows_per_case\":" << windows
          << ",\"gpu_owned_feature_copy\":" << (gpuFeatureCopy ? "true" : "false")
          << ",\"batch_prefill_commands\":" << batchCommands << ",\"sequential_prefill_commands\":" << scalarCommands
          << ",\"rejected_invalid_calls\":" << rejections << ",\"batch_prefill_gpu_seconds\":" << batchGpu
          << ",\"sequential_prefill_gpu_seconds\":" << scalarGpu << ",\"batch_prefill_wall_seconds\":" << batchWall
          << ",\"sequential_prefill_wall_seconds\":" << scalarWall << ",\"comparison\":";
      comparison.write(output); output << ",\"cases\":[";
      for (size_t index = 0; index < records.size(); ++index) { if (index) output << ','; output << records[index]; }
      output << "]}\n";
      if (!output) { std::cerr << "could not write oracle report\n"; return 1; }
    }
    if (!error.empty()) std::cerr << error << '\n';
    if (!comparison.pass) { std::cerr << "batch prefill comparison failed; maximum relative L2=" << comparison.maxL2
        << ", minimum cosine=" << comparison.minimumCosine << ", worst=" << comparison.worst << '\n'; return 1; }
    std::cout << "Flash batch prefill oracle passed " << scenarios << " cases\n";
    return 0;
  }
}
