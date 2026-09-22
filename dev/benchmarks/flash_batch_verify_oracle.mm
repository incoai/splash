// Root-run GPU oracle. --help and --cpu-self-test create no Metal backend.
// Copy every borrowed output before another executor call.
#include "flash/FlashBatchVerify.hpp"
#include "flash/FlashBatchForward.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::CommandTiming;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
constexpr uint32_t kCapacity = 4096, kVocabulary = 248320, kHidden = 10240;
constexpr uint32_t kTrials = 2, kContinuationSteps = 5;
constexpr std::array<const char *, 9> kCanonicalFlags{
    "SPLASH_FLASH_FUSE_HC", "SPLASH_FLASH_FUSE_GDN",
    "SPLASH_FLASH_DENSE_CACHE", "SPLASH_FLASH_BLOCKED_MOE",
    "SPLASH_FLASH_QSA_F32", "SPLASH_FLASH_QSA_MPP",
    "SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_EXPERT_QMV",
    "SPLASH_FLASH_MOE_Q4X8"};
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
std::string hexWord(uint16_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0') << std::setw(4) << value;
  return out.str();
}
template <class T> void writeArray(std::ostream &out, std::span<const T> values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}
template <class T> void writeArray(std::ostream &out, const std::vector<T> &values) {
  writeArray(out, std::span<const T>(values));
}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
struct SHA256 final {
  CC_SHA256_CTX context{};
  SHA256() { CC_SHA256_Init(&context); }
  void add(const void *data, uint64_t bytes) {
    const auto *next = static_cast<const std::byte *>(data);
    while (bytes) {
      const auto count = CC_LONG(std::min<uint64_t>(bytes, std::numeric_limits<CC_LONG>::max()));
      CC_SHA256_Update(&context, next, count); next += count; bytes -= count;
    }
  }
  std::string finish() {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    std::ostringstream out;
    for (auto byte : digest)
      out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
    return out.str();
  }
};
#pragma clang diagnostic pop
template <class T> std::string sha256(std::span<const T> values) {
  SHA256 hash; hash.add(values.data(), values.size_bytes()); return hash.finish();
}
template <class T> std::string sha256(const std::vector<T> &values) {
  return sha256(std::span<const T>(values));
}
std::string fileSHA256(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "could not read hash input: " + path.string());
  SHA256 hash; std::array<char, 1 << 16> bytes{};
  while (input) {
    input.read(bytes.data(), bytes.size());
    if (input.gcount()) hash.add(bytes.data(), uint64_t(input.gcount()));
  }
  require(input.eof(), "hash input read failed: " + path.string());
  return hash.finish();
}
std::filesystem::path executablePath() {
  uint32_t capacity = 4096;
  std::vector<char> bytes(capacity);
  if (_NSGetExecutablePath(bytes.data(), &capacity) != 0) {
    bytes.resize(capacity);
    require(_NSGetExecutablePath(bytes.data(), &capacity) == 0, "could not locate executable");
  }
  return std::filesystem::canonical(bytes.data());
}
std::filesystem::path sourceRoot() {
  if (const char *setting = std::getenv("FLASH_BATCH_VERIFY_SOURCE_ROOT")) {
    const auto root = std::filesystem::canonical(setting);
    require(std::filesystem::is_regular_file(root / "runtime/flash/FlashBatchVerify.hpp"),
            "FLASH_BATCH_VERIFY_SOURCE_ROOT does not contain the batch API");
    return root;
  }
  for (auto candidate : {std::filesystem::current_path(),
                         std::filesystem::absolute(__FILE__).parent_path(),
                         executablePath().parent_path()}) {
    for (;;) {
      if (std::filesystem::is_regular_file(candidate / "runtime/flash/FlashBatchVerify.hpp"))
        return std::filesystem::canonical(candidate);
      const auto parent = candidate.parent_path();
      if (parent == candidate || parent.empty()) break;
      candidate = parent;
    }
  }
  throw std::invalid_argument("cannot locate source tree; set FLASH_BATCH_VERIFY_SOURCE_ROOT");
}
using SourceHashes = std::map<std::string, std::string>;
SourceHashes sourceHashes(const std::filesystem::path &root) {
  SourceHashes result;
  // Executed canonical routes, including the pending AR-batch safety guard.
  // Optional cache/MPP/worker/MTP candidates are excluded. The loaded binary
  // and complete metallib are independently hashed before and after the run.
  for (const char *relative : {
      "dev/benchmarks/flash_batch_verify_oracle.mm", "runtime/engine/Json.hpp",
      "runtime/flash/FlashBatchVerify.hpp", "runtime/flash/FlashBatchVerify.cpp",
      "runtime/flash/FlashBatchVerifyGDN.hpp", "runtime/flash/FlashBatchVerifyGDN.cpp",
      "runtime/flash/FlashForward.hpp", "runtime/flash/FlashForward.cpp",
      "runtime/flash/FlashBatchForward.hpp", "runtime/flash/FlashBatchForward.cpp",
      "runtime/flash/FlashRequestStateInternal.hpp",
      "runtime/flash/FlashDescriptor.hpp", "runtime/flash/FlashDescriptor.mm",
      "runtime/flash/FlashWeights.hpp", "runtime/flash/FlashWeights.mm",
      "runtime/flash/FlashAffine.hpp", "runtime/flash/FlashAffine.cpp",
      "runtime/flash/FlashGDN.hpp", "runtime/flash/FlashGDN.cpp",
      "runtime/flash/FlashGDNFused.hpp", "runtime/flash/FlashGDNFused.cpp",
      "runtime/flash/FlashHC.hpp", "runtime/flash/FlashHC.cpp",
      "runtime/flash/FlashMoE.hpp", "runtime/flash/FlashMoE.cpp",
      "runtime/flash/FlashPLE.hpp", "runtime/flash/FlashPLE.cpp",
      "runtime/flash/FlashQSA.hpp", "runtime/flash/FlashQSA.cpp",
      "runtime/metal/CommandGraph.hpp", "runtime/metal/MetalBackend.hpp",
      "runtime/metal/MetalBackend.mm",
      "runtime/metal/abi/FlashAffine.h", "runtime/metal/abi/FlashGDN.h",
      "runtime/metal/abi/FlashGDNFused.h", "runtime/metal/abi/FlashGDNSeparate.h",
      "runtime/metal/abi/FlashHC.h", "runtime/metal/abi/FlashMoE.h",
      "runtime/metal/abi/FlashPLE.h", "runtime/metal/abi/FlashQSA.h",
      "runtime/metal/abi/FlashForward.h",
      "runtime/metal/kernels/shared/flash_affine.metal",
      "runtime/metal/kernels/shared/flash_gdn.metal",
      "runtime/metal/kernels/shared/flash_gdn_fused.metal",
      "runtime/metal/kernels/shared/flash_hc.metal",
      "runtime/metal/kernels/shared/flash_moe.metal",
      "runtime/metal/kernels/shared/flash_ple.metal",
      "runtime/metal/kernels/shared/flash_qsa.metal",
      "runtime/metal/kernels/shared/flash_forward.metal"}) {
    result.emplace(relative, fileSHA256(root / relative));
  }
  return result;
}
std::string sourceFingerprint(const SourceHashes &sources) {
  SHA256 hash;
  for (const auto &[path, digest] : sources) {
    hash.add(path.data(), path.size()); hash.add("\0", 1);
    hash.add(digest.data(), digest.size()); hash.add("\n", 1);
  }
  return hash.finish();
}
struct CanonicalEnvironment final {
  std::vector<std::pair<std::string, std::optional<std::string>>> previous;
  CanonicalEnvironment() {
    for (const char *name : kCanonicalFlags) {
      const char *value = std::getenv(name);
      previous.emplace_back(name, value ? std::optional<std::string>(value) : std::nullopt);
      if (setenv(name, "0", 1) != 0) {
        restore(); throw std::runtime_error(std::string("could not set ") + name);
      }
    }
  }
  void restore() noexcept {
    for (const auto &[name, value] : previous) {
      if (value) (void)setenv(name.c_str(), value->c_str(), 1);
      else (void)unsetenv(name.c_str());
    }
    previous.clear();
  }
  ~CanonicalEnvironment() { restore(); }
};
std::vector<uint32_t> choices(const char *raw) {
  if (!raw) return {1, 2, 3, 4};
  std::stringstream input(raw); std::vector<uint32_t> result; std::string item;
  while (std::getline(input, item, ',')) {
    require(!item.empty() && item.front() >= '1' && item.front() <= '4', "invalid case selector");
    size_t used = 0; const auto value = std::stoul(item, &used);
    require(used == item.size() && value >= 1 && value <= 4, "case selector must contain 1..4");
    result.push_back(uint32_t(value));
  }
  require(!result.empty() && std::string_view(raw).back() != ',', "empty case selector");
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}
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
    NSNumber *word = static_cast<NSNumber *>(entry); const double value = word.doubleValue;
    require(std::isfinite(value) && value >= 0 && value < kVocabulary && std::floor(value) == value,
            "invalid prompt token");
    result.push_back(uint32_t(value));
  }
  require(!result.empty() && result.size() <= 64, "prompt must contain 1..64 tokens");
  return result;
}
std::vector<uint16_t> copy(const MetalBuffer &buffer, uint64_t elements, const char *label) {
  require(buffer && buffer.contents() && buffer.sizeBytes() >= elements * sizeof(uint16_t),
          std::string(label) + " has invalid extent or CPU visibility");
  std::vector<uint16_t> result(size_t(elements), uint16_t{});
  std::memcpy(result.data(), buffer.contents(), result.size() * sizeof(uint16_t)); return result;
}
struct Snapshot final { std::vector<uint16_t> logits, hidden; };
Snapshot snapshot(const FlashForwardResult &result, uint32_t rows) {
  require(result.logitRows == rows, "scalar output row count differs");
  return {copy(result.logitsBF16, uint64_t(rows) * kVocabulary, "scalar logits"),
          copy(result.hiddenBF16, uint64_t(rows) * kHidden, "scalar hidden")};
}
Snapshot snapshot(const FlashBatchVerifyResult &result, uint32_t lanes, uint32_t rows) {
  require(result.lanes == lanes && result.rows == rows && result.capacity == kCapacity &&
          result.logicalLengths.size() == lanes, "batch output geometry differs");
  return {copy(result.logitsBF16, uint64_t(lanes) * rows * kVocabulary, "batch logits"),
          copy(result.hiddenBF16, uint64_t(lanes) * rows * kHidden, "batch hidden")};
}
struct Mismatch final { uint32_t lane, row, column; uint16_t actual, expected; };
struct Error final {
  uint64_t elements = 0, mismatch = 0, nonfinite = 0; uint32_t maximumULP = 0;
  double maximumAbsolute = 0, squaredError = 0, squaredReference = 0;
  std::vector<Mismatch> first, firstNonfinite;
  bool pass() const { return !mismatch && !nonfinite; }
  void add(std::span<const uint16_t> actual, std::span<const uint16_t> expected,
           uint32_t lane, uint32_t width) {
    require(actual.size() == expected.size() && actual.size() % width == 0, "comparison extents differ");
    for (size_t i = 0; i < actual.size(); ++i) {
      ++elements; const uint16_t a = actual[i], b = expected[i];
      const float x = number(a), y = number(b);
      if (a != b) {
        ++mismatch;
        if (first.size() < 8) first.push_back({lane, uint32_t(i / width), uint32_t(i % width), a, b});
      }
      if (!std::isfinite(x) || !std::isfinite(y)) {
        ++nonfinite;
        if (firstNonfinite.size() < 8)
          firstNonfinite.push_back({lane, uint32_t(i / width), uint32_t(i % width), a, b});
        continue;
      }
      const double delta = double(x) - y;
      maximumAbsolute = std::max(maximumAbsolute, std::abs(delta));
      squaredError += delta * delta; squaredReference += double(y) * y;
      const auto ordered = [](uint16_t value) -> uint32_t {
        return (value & 0x8000) ? uint32_t(0x8000 - (value & 0x7fff)) : uint32_t(0x8000 + value);
      };
      const auto u = ordered(a), v = ordered(b);
      maximumULP = std::max(maximumULP, u > v ? u - v : v - u);
    }
  }
  void write(std::ostream &out) const {
    out << "{\"pass\":" << (pass() ? "true" : "false") << ",\"elements\":" << elements
        << ",\"bf16_mismatches\":" << mismatch << ",\"nonfinite\":" << nonfinite
        << ",\"max_bf16_ulp\":" << maximumULP << ",\"max_abs\":" << maximumAbsolute
        << ",\"relative_l2\":" << std::sqrt(squaredError / std::max(1e-30, squaredReference));
    const auto writeFirst = [&](const char *name, const std::vector<Mismatch> &values) {
      out << ',' << splash::json::quote(name) << ":[";
      for (size_t i = 0; i < values.size(); ++i) {
      if (i) out << ','; const auto &value = values[i];
      out << "{\"lane\":" << value.lane << ",\"row\":" << value.row << ",\"column\":" << value.column
          << ",\"actual_bf16\":" << splash::json::quote(hexWord(value.actual))
          << ",\"expected_bf16\":" << splash::json::quote(hexWord(value.expected)) << ",\"actual\":";
      if (std::isfinite(number(value.actual))) out << number(value.actual); else out << "null";
      out << ",\"expected\":";
      if (std::isfinite(number(value.expected))) out << number(value.expected); else out << "null";
      out << '}';
      }
      out << ']';
    };
    writeFirst("first_mismatches", first); writeFirst("first_nonfinite", firstNonfinite); out << '}';
  }
};
uint32_t greedyOrInvalid(std::span<const uint16_t> logits) {
  require(!logits.empty(), "empty logits"); uint32_t best = 0;
  float maximum = -std::numeric_limits<float>::infinity();
  for (uint32_t token = 0; token < logits.size(); ++token) {
    const float score = number(logits[token]);
    if (!std::isfinite(score)) return std::numeric_limits<uint32_t>::max();
    if (score > maximum) { maximum = score; best = token; }
  }
  return best;
}
uint32_t greedy(std::span<const uint16_t> logits) {
  const auto token = greedyOrInvalid(logits);
  require(token != std::numeric_limits<uint32_t>::max(), "nonfinite logits during greedy selection"); return token;
}
uint32_t continuationAnchor(std::span<const uint16_t> logits) {
  // Numerical diagnostics already mark this case failed. A legal common
  // fallback token lets the case finish and preserve all failing coordinates.
  const auto token = greedyOrInvalid(logits);
  return token == std::numeric_limits<uint32_t>::max() ? 0 : token;
}
std::span<const uint16_t> laneValues(const std::vector<uint16_t> &values,
                                   uint32_t lane, uint32_t rows, uint32_t width) {
  return std::span<const uint16_t>(values).subspan(uint64_t(lane) * rows * width, uint64_t(rows) * width);
}
struct Comparison final {
  Error logits, hidden; uint64_t greedyMismatch = 0;
  std::vector<uint32_t> actualGreedy, expectedGreedy;
  bool pass() const { return logits.pass() && hidden.pass() && !greedyMismatch; }
  void add(const Snapshot &actual, const Snapshot &expected, uint32_t lane,
           uint32_t rows, bool actualIsBatch = true) {
    const auto a = laneValues(actual.logits, actualIsBatch ? lane : 0, rows, kVocabulary);
    logits.add(a, expected.logits, lane, kVocabulary);
    hidden.add(laneValues(actual.hidden, actualIsBatch ? lane : 0, rows, kHidden), expected.hidden, lane, kHidden);
    for (uint32_t row = 0; row < rows; ++row) {
      const auto x = greedyOrInvalid(a.subspan(uint64_t(row) * kVocabulary, kVocabulary));
      const auto y = greedyOrInvalid(std::span<const uint16_t>(expected.logits).subspan(uint64_t(row) * kVocabulary, kVocabulary));
      actualGreedy.push_back(x); expectedGreedy.push_back(y); greedyMismatch += x != y;
    }
  }
  void write(std::ostream &out) const {
    out << "{\"pass\":" << (pass() ? "true" : "false") << ",\"logits\":"; logits.write(out);
    out << ",\"hidden\":"; hidden.write(out);
    out << ",\"greedy_mismatches\":" << greedyMismatch
        << ",\"nonfinite_greedy_sentinel\":" << std::numeric_limits<uint32_t>::max()
        << ",\"actual_greedy\":"; writeArray(out, actualGreedy);
    out << ",\"expected_greedy\":"; writeArray(out, expectedGreedy); out << '}';
  }
};
struct Timings final {
  struct Entry { uint64_t commands = 0; double gpu = 0, wall = 0; };
  std::map<std::string, Entry> phases;
  void add(const std::string &phase, CommandTiming timing, bool command = true) {
    require(std::isfinite(timing.gpuSeconds) && std::isfinite(timing.wallSeconds) &&
            timing.gpuSeconds >= 0 && timing.wallSeconds >= 0, "invalid command timing");
    auto &value = phases[phase]; value.commands += command;
    value.gpu += timing.gpuSeconds; value.wall += timing.wallSeconds;
  }
  void write(std::ostream &out) const {
    Entry total; out << "{\"phases\":{"; bool comma = false;
    for (const auto &[phase, value] : phases) {
      if (comma) out << ','; comma = true;
      out << splash::json::quote(phase) << ":{\"commands\":" << value.commands
          << ",\"gpu_seconds\":" << value.gpu << ",\"wall_seconds\":" << value.wall << '}';
      total.commands += value.commands; total.gpu += value.gpu; total.wall += value.wall;
    }
    out << "},\"accounted_successful_gpu_commands\":" << total.commands
        << ",\"command_count_basis\":\"successful_synchronous_executor_returns\""
        << ",\"gpu_seconds\":" << total.gpu << ",\"wall_seconds\":" << total.wall << '}';
  }
};
struct Report final {
  bool pass = true, complete = false, sourceStable = false, inputsStable = false;
  std::string failure, sourceIdentity, manifestFingerprint, kernelRoutes;
  std::filesystem::path root; SourceHashes sources;
  std::map<std::string, std::string> inputs;
  std::map<std::string, std::filesystem::path> inputPaths; uint64_t trunkBytes = 0, batchBytes = 0;
  std::vector<uint32_t> lanes, rows; std::vector<std::string> cases, safety; Timings timings;
  void write(const char *path) const {
    std::ofstream out(path); require(bool(out), "could not open report");
    out << std::setprecision(17) << "{\"schema\":\"splash-flash-batch-verify-oracle-v1\",\"pass\":"
        << (pass && complete && sourceStable && inputsStable ? "true" : "false")
        << ",\"complete\":" << (complete ? "true" : "false") << ",\"failure\":" << splash::json::quote(failure)
        << ",\"source_identity\":" << splash::json::quote(sourceIdentity)
        << ",\"manifest_fingerprint\":" << splash::json::quote(manifestFingerprint)
        << ",\"payload_hashes_verified\":false,\"source_root\":" << splash::json::quote(root.string())
        << ",\"source_stability_scope\":\"explicit_canonical_operator_reference_dependencies\""
        << ",\"source_files_stable_during_run\":" << (sourceStable ? "true" : "false")
        << ",\"input_files_stable_during_run\":" << (inputsStable ? "true" : "false")
        << ",\"source_tree_sha256\":" << splash::json::quote(sourceFingerprint(sources))
        << ",\"source_files_sha256\":{"; bool comma = false;
    for (const auto &[name, digest] : sources) {
      if (comma) out << ','; comma = true; out << splash::json::quote(name) << ':' << splash::json::quote(digest);
    }
    out << "},\"input_files_sha256\":{"; comma = false;
    for (const auto &[name, digest] : inputs) {
      if (comma) out << ','; comma = true; out << splash::json::quote(name) << ':' << splash::json::quote(digest);
    }
    out << "},\"input_files\":{"; comma = false;
    for (const auto &[name, path] : inputPaths) {
      if (comma) out << ','; comma = true; out << splash::json::quote(name) << ':' << splash::json::quote(path.string());
    }
    out << "},\"forward_semantics\":" << splash::json::quote(kFlashForwardSemantics)
        << ",\"batch_verify_semantics\":" << splash::json::quote(kFlashBatchVerifySemantics)
        << ",\"kernel_routes\":" << splash::json::quote(kernelRoutes) << ",\"canonical_flags\":{"; comma = false;
    for (const char *name : kCanonicalFlags) {
      if (comma) out << ','; comma = true; out << splash::json::quote(name) << ":\"0\"";
    }
    out << "},\"capacity\":" << kCapacity << ",\"trunk_workspace_bytes\":" << trunkBytes
        << ",\"batch_workspace_bytes\":" << batchBytes << ",\"trials_per_case\":" << kTrials
        << ",\"continuation_steps\":" << kContinuationSteps << ",\"lanes\":"; writeArray(out, lanes);
    out << ",\"rows\":"; writeArray(out, rows);
    out << ",\"full_width_row_matrix\":" << (lanes.size() == 4 && rows.size() == 4 ? "true" : "false")
        << ",\"qsa_sparse_top512_selection_qualified\":false,\"timing\":"; timings.write(out);
    out << ",\"cases\":[";
    for (size_t i = 0; i < cases.size(); ++i) { if (i) out << ','; out << cases[i]; }
    out << "],\"safety_checks\":[";
    for (size_t i = 0; i < safety.size(); ++i) { if (i) out << ','; out << safety[i]; }
    out << "]}\n"; require(bool(out), "could not write report");
  }
};
std::vector<FlashRequestState *> pointers(std::vector<FlashRequestState> &states) {
  std::vector<FlashRequestState *> result;
  for (auto &state : states) result.push_back(&state); return result;
}
std::vector<FlashRequestState> fresh(FlashForward &trunk, uint32_t count) {
  std::vector<FlashRequestState> result; result.reserve(count);
  for (uint32_t lane = 0; lane < count; ++lane) result.push_back(trunk.createState()); return result;
}
std::vector<uint32_t> prompt(const std::vector<uint32_t> &tokens, uint32_t lane) {
  std::vector<uint32_t> result; const size_t length = std::max<size_t>(8, tokens.size()) + lane;
  for (size_t row = 0; row < length; ++row) result.push_back(tokens[(row + lane) % tokens.size()]);
  return result;
}
uint32_t prefill(FlashForward &trunk, FlashRequestState &state, const std::vector<uint32_t> &tokens,
                 Report &report, const char *phase) {
  const auto result = trunk.forward(state, tokens);
  const auto logits = copy(result.logitsBF16, kVocabulary, "prefill logits"); report.timings.add(phase, result.timing);
  require(result.logicalLength == tokens.size() && state.logicalLength() == tokens.size() && !state.poisoned(),
          "prefill state metadata differs"); return greedy(logits);
}
Snapshot retainedOutput(const Snapshot &output, uint32_t lane, uint32_t rows, uint32_t retained) {
  const auto logits = laneValues(output.logits, lane, rows, kVocabulary).first(uint64_t(retained) * kVocabulary);
  const auto hidden = laneValues(output.hidden, lane, rows, kHidden).first(uint64_t(retained) * kHidden);
  return {{logits.begin(), logits.end()}, {hidden.begin(), hidden.end()}};
}
void runCase(FlashForward &trunk, FlashBatchVerify &batch, uint32_t lanes, uint32_t rows,
             const std::vector<uint32_t> &tokens, Report &report) {
  auto actual = fresh(trunk, lanes), reference = fresh(trunk, lanes);
  const bool independent = lanes == 4 && rows == 4;
  auto prefixReference = fresh(trunk, independent ? lanes : 0);
  auto active = pointers(actual);
  std::vector<uint32_t> anchors(lanes); std::vector<uint64_t> prefixLengths;
  std::vector<std::string> promptHashes;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto input = prompt(tokens, lane);
    const auto a = prefill(trunk, actual[lane], input, report, "batch_state_prefill");
    anchors[lane] = prefill(trunk, reference[lane], input, report, "scalar_state_prefill");
    require(a == anchors[lane], "fresh paired prefill greedy differs");
    if (independent)
      require(prefill(trunk, prefixReference[lane], input, report, "independent_state_prefill") == a,
              "independent prefill greedy differs");
    prefixLengths.push_back(input.size()); promptHashes.push_back(sha256(input));
  }
  std::ostringstream record; record << std::setprecision(17);
  record << "{\"lanes\":" << lanes << ",\"rows\":" << rows << ",\"prefix_lengths\":"; writeArray(record, prefixLengths);
  record << ",\"expanded_prompt_u32_native_sha256\":[";
  for (size_t i = 0; i < promptHashes.size(); ++i) { if (i) record << ','; record << splash::json::quote(promptHashes[i]); }
  record << "],\"independent_retained_prefix_reference\":" << (independent ? "true" : "false") << ",\"trials\":[";
  bool casePass = true;
  for (uint32_t trial = 0; trial < kTrials; ++trial) {
    std::vector<uint32_t> incoming, retained(lanes); std::vector<uint64_t> begins;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      begins.push_back(actual[lane].logicalLength());
      require(begins.back() == reference[lane].logicalLength(), "paired begin length differs");
      retained[lane] = trial == 0 ? 1 + lane % rows : rows - lane % rows;
      incoming.push_back(anchors[lane]);
      for (uint32_t row = 1; row < rows; ++row)
        incoming.push_back(tokens[(lane * 7 + row * 3 + trial) % tokens.size()]);
    }
    const auto result = batch.verifyBatch(active, incoming, rows);
    const auto output = snapshot(result, lanes, rows); report.timings.add("batch_verify", result.timing);
    require(batch.pending(), "batch trial was not marked pending");
    Comparison compared, independentCompared;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(result.logicalLengths[lane] == begins[lane] + rows && actual[lane].logicalLength() == begins[lane] + rows &&
              !actual[lane].poisoned() && !trunk.ownsState(actual[lane]), "provisional batch metadata differs");
      const auto input = std::span<const uint32_t>(incoming).subspan(uint64_t(lane) * rows, rows);
      const auto scalar = trunk.verify(reference[lane], input);
      const auto expected = snapshot(scalar, rows); report.timings.add("scalar_verify", scalar.timing);
      require(scalar.logicalLength == begins[lane] + rows, "scalar trial length differs");
      compared.add(output, expected, lane, rows);
      anchors[lane] = continuationAnchor(std::span<const uint16_t>(expected.logits).subspan(
          uint64_t(retained[lane] - 1) * kVocabulary, kVocabulary));
      const auto commit = trunk.commitVerify(reference[lane], retained[lane]);
      if (retained[lane] == rows)
        require(commit.gpuSeconds == 0 && commit.wallSeconds == 0, "scalar full commit submitted work");
      report.timings.add("scalar_commit", commit, retained[lane] != rows);
      if (independent) {
        const auto prefix = trunk.forward(prefixReference[lane], input.first(retained[lane]), true, true);
        const auto expectedPrefix = snapshot(prefix, retained[lane]); report.timings.add("independent_retained_prefix", prefix.timing);
        independentCompared.add(retainedOutput(output, lane, rows, retained[lane]), expectedPrefix, lane, retained[lane], false);
      }
    }
    const bool allFull = std::all_of(retained.begin(), retained.end(), [rows](auto count) { return count == rows; });
    const auto commit = batch.commitBatch(active, retained); report.timings.add("batch_commit", commit, !allFull);
    if (allFull) require(commit.gpuSeconds == 0 && commit.wallSeconds == 0, "batch full commit submitted work");
    require(!batch.pending(), "valid batch commit left a pending trial");
    std::vector<uint64_t> committed;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      committed.push_back(actual[lane].logicalLength());
      require(committed.back() == begins[lane] + retained[lane] && committed.back() == reference[lane].logicalLength() &&
              trunk.ownsState(actual[lane]), "committed prefix metadata differs");
      if (independent) require(committed.back() == prefixReference[lane].logicalLength(), "independent prefix length differs");
    }
    casePass &= compared.pass() && (!independent || independentCompared.pass());
    if (trial) record << ',';
    record << "{\"trial\":" << trial << ",\"begin_lengths\":"; writeArray(record, begins);
    record << ",\"incoming_lane_major_tokens\":"; writeArray(record, incoming);
    record << ",\"retained\":"; writeArray(record, retained); record << ",\"committed_lengths\":"; writeArray(record, committed);
    record << ",\"batch_logits_sha256\":" << splash::json::quote(sha256(output.logits))
           << ",\"batch_hidden_sha256\":" << splash::json::quote(sha256(output.hidden)) << ",\"comparison\":"; compared.write(record);
    if (independent) { record << ",\"ordinary_retained_prefix_reference\":"; independentCompared.write(record); }
    record << '}';
  }
  record << "],\"continuation\":[";
  for (uint32_t step = 0; step < kContinuationSteps; ++step) {
    Comparison compared, independentCompared; std::vector<uint32_t> input(lanes); std::vector<uint64_t> lengths;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      // Common forced input preserves a meaningful comparison even if an
      // incorrect executor changes its argmax, and replaces provisional tails.
      input[lane] = (anchors[lane] + 1 + 17 * lane + 31 * step) % kVocabulary;
      const auto a = trunk.forward(actual[lane], std::span<const uint32_t>(&input[lane], 1), true, true);
      const auto output = snapshot(a, 1); report.timings.add("batch_state_ar_continuation", a.timing);
      const auto b = trunk.forward(reference[lane], std::span<const uint32_t>(&input[lane], 1), true, true);
      const auto expected = snapshot(b, 1); report.timings.add("scalar_state_ar_continuation", b.timing);
      compared.add(output, expected, lane, 1, false); anchors[lane] = continuationAnchor(expected.logits);
      lengths.push_back(actual[lane].logicalLength());
      require(lengths.back() == reference[lane].logicalLength() && trunk.ownsState(actual[lane]), "AR continuation length differs");
      if (independent) {
        const auto c = trunk.forward(prefixReference[lane], std::span<const uint32_t>(&input[lane], 1), true, true);
        const auto expectedPrefix = snapshot(c, 1); report.timings.add("independent_state_ar_continuation", c.timing);
        independentCompared.add(output, expectedPrefix, lane, 1, false);
        require(lengths.back() == prefixReference[lane].logicalLength(), "independent AR length differs");
      }
    }
    casePass &= compared.pass() && (!independent || independentCompared.pass());
    if (step) record << ',';
    record << "{\"step\":" << step << ",\"incoming_tokens\":"; writeArray(record, input);
    record << ",\"logical_lengths\":"; writeArray(record, lengths); record << ",\"scalar_verify_commit_reference\":"; compared.write(record);
    if (independent) { record << ",\"ordinary_retained_prefix_reference\":"; independentCompared.write(record); }
    record << '}';
  }
  record << "],\"pass\":" << (casePass ? "true" : "false") << '}';
  report.cases.push_back(record.str()); report.pass &= casePass;
  std::cerr << "flash-batch-verify-oracle lanes=" << lanes << " rows=" << rows << " pass=" << (casePass ? "true" : "false") << '\n';
}
template <class Function> void rejects(Report &report, const char *name, Function action) {
  std::string message; bool rejected = false;
  try { action(); } catch (const std::exception &error) { rejected = true; message = error.what(); }
  report.safety.push_back("{\"name\":" + splash::json::quote(name) + ",\"pass\":" + (rejected ? "true" : "false") +
                          ",\"rejection\":" + splash::json::quote(message) + '}');
  report.pass &= rejected; require(rejected, std::string("expected rejection: ") + name);
}
void records(Report &report, const char *name) {
  report.safety.push_back("{\"name\":" + splash::json::quote(name) + ",\"pass\":true}");
}
void runSafety(MetalBackend &backend, const FlashWeights &weights, FlashForward &trunk,
               FlashBatchVerify &batch, uint32_t token, Report &report) {
  auto states = fresh(trunk, 5); auto active = pointers(states); std::vector<uint32_t> incoming(20, token);
  const auto pair = std::span<FlashRequestState *const>(active).first(2), single = pair.first(1);
  const auto input = std::span<const uint32_t>(incoming);
  const auto verify = [&](std::span<FlashRequestState *const> selection, std::span<const uint32_t> frame, uint32_t rows) {
    (void)batch.verifyBatch(selection, frame, rows);
  };
  const auto coldUnchanged = [&] {
    require(!batch.pending(), "rejected cold trial became pending");
    for (const auto &state : states)
      require(state.logicalLength() == 0 && !state.poisoned() && trunk.ownsState(state), "rejected cold trial mutated a state");
  };
  rejects(report, "empty_lanes", [&] { verify({}, {}, 1); }); coldUnchanged();
  rejects(report, "too_many_lanes", [&] { verify(active, input.first(5), 1); }); coldUnchanged();
  rejects(report, "rows_zero", [&] { verify(pair, {}, 0); }); coldUnchanged();
  rejects(report, "rows_five", [&] { verify(pair, input.first(10), 5); }); coldUnchanged();
  rejects(report, "token_count_short", [&] { verify(pair, input.first(3), 2); }); coldUnchanged();
  rejects(report, "token_count_long", [&] { verify(pair, input.first(5), 2); }); coldUnchanged();
  std::array<FlashRequestState *, 2> duplicate{active[0], active[0]}, nullLane{active[0], nullptr};
  rejects(report, "duplicate_state", [&] { verify(duplicate, input.first(4), 2); }); coldUnchanged();
  rejects(report, "later_null_state", [&] { verify(nullLane, input.first(4), 2); }); coldUnchanged();
  FlashRequestState uninitialized; std::array<FlashRequestState *, 2> defaultLane{active[0], &uninitialized};
  rejects(report, "later_default_state", [&] { verify(defaultLane, input.first(4), 2); }); coldUnchanged();
  const auto invalidTokens = std::vector<uint32_t>{token, token, token, kVocabulary};
  rejects(report, "later_lane_last_token_oov", [&] { verify(pair, invalidTokens, 2); }); coldUnchanged();
  rejects(report, "constructor_context_mismatch", [&] { FlashBatchVerify invalid(backend, weights, trunk, kCapacity - 1); }); coldUnchanged();
  for (const auto geometry : std::array<std::array<uint32_t, 3>, 6>{
         {{0, 4, 4}, {262145, 4, 4}, {4, 0, 4}, {4, 5, 4}, {4, 4, 0}, {4, 4, 5}}}) {
    const auto name = "planned_bytes_invalid_" + std::to_string(geometry[0]) + "_" +
                      std::to_string(geometry[1]) + "_" + std::to_string(geometry[2]);
    rejects(report, name.c_str(), [&] { (void)FlashBatchVerify::workspacePlannedBytes(geometry[0], geometry[1], geometry[2]); });
    coldUnchanged();
  }
  {
    FlashForward foreign(backend, weights, 4, 4, 4); auto foreignState = foreign.createState();
    std::array<FlashRequestState *, 2> foreignLane{active[0], &foreignState};
    rejects(report, "later_foreign_owner_state", [&] { verify(foreignLane, input.first(4), 2); }); coldUnchanged();
    FlashBatchVerify bounded(backend, weights, foreign, 4); auto contextStates = fresh(foreign, 2);
    const std::array<uint32_t, 3> prefix{token, token, token};
    const auto result = foreign.forward(contextStates[1], prefix);
    (void)copy(result.logitsBF16, kVocabulary, "context safety prefill logits");
    report.timings.add("safety_context_prefill", result.timing); auto contextPointers = pointers(contextStates);
    rejects(report, "later_lane_context_overflow", [&] { (void)bounded.verifyBatch(contextPointers, input.first(4), 2); });
    require(!bounded.pending() && contextStates[0].logicalLength() == 0 && contextStates[1].logicalLength() == 3 &&
            !contextStates[0].poisoned() && !contextStates[1].poisoned(), "context rejection mutated a lane");
  }
  {
    auto scalarState = trunk.createState(); const auto result = trunk.verify(scalarState, input.first(2));
    (void)snapshot(result, 2); report.timings.add("safety_scalar_verify", result.timing);
    std::array<FlashRequestState *, 2> scalarPending{active[0], &scalarState};
    rejects(report, "later_scalar_pending_state", [&] { verify(scalarPending, input.first(4), 2); }); coldUnchanged();
    require(scalarState.logicalLength() == 2 && !scalarState.poisoned(), "scalar pending rejection mutated state");
    report.timings.add("safety_scalar_commit", trunk.commitVerify(scalarState, 1));
  }
  rejects(report, "commit_without_pending", [&] { const std::array<uint32_t, 2> counts{1, 1}; (void)batch.commitBatch(pair, counts); }); coldUnchanged();
  const auto trial = batch.verifyBatch(pair, input.first(4), 2);
  const auto before = snapshot(trial, 2, 2); report.timings.add("safety_batch_verify", trial.timing);
  const auto pendingUnchanged = [&] {
    require(batch.pending(), "rejected pending action cleared trial");
    for (uint32_t lane = 0; lane < 2; ++lane)
      require(states[lane].logicalLength() == 2 && !states[lane].poisoned() && !trunk.ownsState(states[lane]),
              "rejected pending action mutated a lane");
    require(copy(trial.logitsBF16, before.logits.size(), "pending logits") == before.logits &&
            copy(trial.hiddenBF16, before.hidden.size(), "pending hidden") == before.hidden,
            "rejected pending action overwrote borrowed output");
  };
  rejects(report, "verify_while_pending", [&] { verify(single, input.first(1), 1); }); pendingUnchanged();
  rejects(report, "scalar_ar_of_batch_pending_lane", [&] { (void)trunk.forward(states[1], input.first(1)); }); pendingUnchanged();
  rejects(report, "scalar_verify_of_batch_pending_lane", [&] { (void)trunk.verify(states[1], input.first(1)); }); pendingUnchanged();
  {
    FlashBatchForward ar(backend, weights, trunk, kCapacity);
    rejects(report, "ar_batch_of_pending_lanes", [&] { (void)ar.forwardBatch(pair, input.first(2)); }); pendingUnchanged();
  }
  for (const auto counts : {std::array<uint32_t, 2>{1, 3}, std::array<uint32_t, 2>{3, 1},
                            std::array<uint32_t, 2>{0, 3}}) {
    const auto name = "invalid_retained_" + std::to_string(counts[0]) + "_" + std::to_string(counts[1]);
    rejects(report, name.c_str(), [&] { (void)batch.commitBatch(pair, counts); }); pendingUnchanged();
  }
  rejects(report, "commit_short_count_extent", [&] { const std::array<uint32_t, 1> counts{1}; (void)batch.commitBatch(pair, counts); }); pendingUnchanged();
  rejects(report, "commit_long_count_extent", [&] { const std::array<uint32_t, 3> counts{1, 1, 1}; (void)batch.commitBatch(pair, counts); }); pendingUnchanged();
  const std::array<uint32_t, 2> full{2, 2};
  rejects(report, "commit_wrong_lane_extent", [&] { (void)batch.commitBatch(single, full); }); pendingUnchanged();
  const std::array<FlashRequestState *, 2> reordered{active[1], active[0]}, substitute{active[0], active[2]};
  rejects(report, "commit_reordered_lanes", [&] { (void)batch.commitBatch(reordered, full); }); pendingUnchanged();
  rejects(report, "commit_substitute_state", [&] { (void)batch.commitBatch(substitute, full); }); pendingUnchanged();
  rejects(report, "commit_duplicate_state", [&] { (void)batch.commitBatch(duplicate, full); }); pendingUnchanged();
  rejects(report, "commit_nonzero_retained_with_null_state", [&] { (void)batch.commitBatch(nullLane, full); }); pendingUnchanged();
  auto moved = std::move(states[0]);
  require(!trunk.ownsState(states[0]) && moved.logicalLength() == 2 && !moved.poisoned(), "pending wrapper move lost state");
  std::array<FlashRequestState *, 2> movedPointers{&moved, &states[1]};
  const auto committed = batch.commitBatch(movedPointers, full); report.timings.add("safety_full_commit", committed, false);
  require(committed.gpuSeconds == 0 && committed.wallSeconds == 0 && !batch.pending() &&
          trunk.ownsState(moved) && trunk.ownsState(states[1]), "moved pending full commit differs");
  records(report, "pending_wrapper_move_commits_same_impl_with_zero_timing");
  rejects(report, "moved_from_state", [&] { verify(single, input.first(1), 1); });
  {
    auto destroyed = std::make_unique<FlashRequestState>(trunk.createState());
    auto survivor = trunk.createState(), another = trunk.createState();
    std::array<FlashRequestState *, 3> live{destroyed.get(), &survivor, &another};
    const auto result = batch.verifyBatch(live, input.first(6), 2);
    (void)snapshot(result, 3, 2); report.timings.add("safety_abort_verify", result.timing);
    destroyed.reset(); auto movedSurvivor = std::move(survivor);
    batch.abortBatch(); batch.abortBatch();
    require(!batch.pending() && movedSurvivor.poisoned() && another.poisoned() &&
            !trunk.ownsState(movedSurvivor) && !trunk.ownsState(another),
            "abort failed to poison every live survivor after wrapper destruction/move");
    records(report, "abort_after_pending_lane_destruction_and_survivor_move_is_terminal_and_idempotent");
    std::array<FlashRequestState *, 1> poisoned{&movedSurvivor};
    rejects(report, "aborted_state_scalar_ar", [&] { (void)trunk.forward(movedSurvivor, input.first(1)); });
    rejects(report, "aborted_state_batch_verify", [&] { verify(poisoned, input.first(1), 1); });
    require(!batch.pending() && movedSurvivor.poisoned(), "terminal rejection revived aborted state");
  }
  {
    auto destroyed = std::make_unique<FlashRequestState>(trunk.createState()); auto survivor = trunk.createState();
    std::array<FlashRequestState *, 2> live{destroyed.get(), &survivor};
    const auto result = batch.verifyBatch(live, input.first(4), 2);
    (void)snapshot(result, 2, 2); report.timings.add("safety_destroyed_cancel_verify", result.timing);
    destroyed.reset(); auto movedSurvivor = std::move(survivor);
    const std::array<FlashRequestState *, 2> resolve{nullptr, &movedSurvivor};
    const std::array<uint32_t, 2> counts{0, 1};
    report.timings.add("safety_destroyed_cancel_commit", batch.commitBatch(resolve, counts));
    require(!batch.pending() && movedSurvivor.logicalLength() == 1 && trunk.ownsState(movedSurvivor) && !movedSurvivor.poisoned(),
            "cancelled destroyed lane prevented healthy survivor commit");
    records(report, "zero_retention_accepts_destroyed_null_lane_and_restores_moved_survivor");
  }
  {
    auto first = trunk.createState(), second = trunk.createState(); std::array<FlashRequestState *, 2> live{&first, &second};
    auto temporary = std::make_unique<FlashBatchVerify>(backend, weights, trunk, kCapacity);
    const auto result = temporary->verifyBatch(live, input.first(4), 2);
    (void)snapshot(result, 2, 2); report.timings.add("safety_destructor_verify", result.timing); temporary.reset();
    require(first.poisoned() && second.poisoned() && !trunk.ownsState(first) && !trunk.ownsState(second),
            "pending executor destruction did not cancel both live lanes");
    records(report, "pending_executor_destruction_poisons_all_live_lanes");
  }
}
void runCancellation(FlashForward &trunk, FlashBatchVerify &batch,
                     const std::vector<uint32_t> &tokens, Report &report) {
  auto actual = fresh(trunk, 2), reference = fresh(trunk, 1), independent = fresh(trunk, 1);
  auto active = pointers(actual); std::array<uint32_t, 2> anchors{};
  for (uint32_t lane = 0; lane < 2; ++lane)
    anchors[lane] = prefill(trunk, actual[lane], prompt(tokens, lane), report, "cancel_batch_state_prefill");
  const auto peerPrompt = prompt(tokens, 1);
  require(prefill(trunk, reference[0], peerPrompt, report, "cancel_scalar_state_prefill") == anchors[1] &&
          prefill(trunk, independent[0], peerPrompt, report, "cancel_independent_state_prefill") == anchors[1],
          "cancel reference prefill differs");
  const uint64_t begin = actual[1].logicalLength();
  std::vector<uint32_t> input;
  for (uint32_t lane = 0; lane < 2; ++lane) {
    input.push_back(anchors[lane]);
    for (uint32_t row = 1; row < 4; ++row) input.push_back(tokens[(row + 3 * lane) % tokens.size()]);
  }
  const auto trial = batch.verifyBatch(active, input, 4);
  const auto output = snapshot(trial, 2, 4); report.timings.add("cancel_batch_verify", trial.timing);
  const auto peerInput = std::span<const uint32_t>(input).subspan(4, 4);
  const auto scalar = trunk.verify(reference[0], peerInput);
  const auto expected = snapshot(scalar, 4); report.timings.add("cancel_scalar_verify", scalar.timing);
  Comparison provisional; provisional.add(output, expected, 1, 4);
  report.timings.add("cancel_scalar_commit", trunk.commitVerify(reference[0], 2));
  const auto prefix = trunk.forward(independent[0], peerInput.first(2), true, true);
  const auto expectedPrefix = snapshot(prefix, 2); report.timings.add("cancel_independent_prefix", prefix.timing);
  Comparison prefixCompared; prefixCompared.add(retainedOutput(output, 1, 4, 2), expectedPrefix, 1, 2, false);
  const std::array<FlashRequestState *, 2> resolve{nullptr, &actual[1]};
  const std::array<uint32_t, 2> retained{0, 2};
  report.timings.add("cancel_batch_commit", batch.commitBatch(resolve, retained));
  require(!batch.pending() && actual[0].poisoned() && !trunk.ownsState(actual[0]) &&
          actual[1].logicalLength() == begin + 2 && trunk.ownsState(actual[1]) && !actual[1].poisoned(),
          "mixed cancellation did not leave exactly one healthy committed peer");
  records(report, "mixed_retention_zero_two_poisons_live_null_lane_and_restores_peer");
  rejects(report, "cancelled_live_null_lane_cannot_ar", [&] { (void)trunk.forward(actual[0], peerInput.first(1)); });
  bool pass = provisional.pass() && prefixCompared.pass();
  std::ostringstream record; record << std::setprecision(17)
    << "{\"kind\":\"mixed_terminal_cancel\",\"lanes\":2,\"rows\":4,\"retained\":[0,2],\"incoming_lane_major_tokens\":";
  writeArray(record, input); record << ",\"survivor_begin_length\":" << begin << ",\"provisional_comparison\":"; provisional.write(record);
  record << ",\"ordinary_retained_prefix_reference\":"; prefixCompared.write(record); record << ",\"continuation\":[";
  uint32_t anchor = continuationAnchor(std::span<const uint16_t>(expected.logits).subspan(kVocabulary, kVocabulary));
  for (uint32_t step = 0; step < kContinuationSteps; ++step) {
    const uint32_t token = (anchor + 19 + 37 * step) % kVocabulary;
    const auto a = trunk.forward(actual[1], std::span<const uint32_t>(&token, 1), true, true);
    const auto continued = snapshot(a, 1); report.timings.add("cancel_survivor_ar", a.timing);
    const auto b = trunk.forward(reference[0], std::span<const uint32_t>(&token, 1), true, true);
    const auto scalarExpected = snapshot(b, 1); report.timings.add("cancel_scalar_ar", b.timing);
    const auto c = trunk.forward(independent[0], std::span<const uint32_t>(&token, 1), true, true);
    const auto prefixExpected = snapshot(c, 1); report.timings.add("cancel_independent_ar", c.timing);
    Comparison compared, prefixContinuation;
    compared.add(continued, scalarExpected, 1, 1, false); prefixContinuation.add(continued, prefixExpected, 1, 1, false);
    pass &= compared.pass() && prefixContinuation.pass(); anchor = continuationAnchor(scalarExpected.logits);
    require(actual[1].logicalLength() == begin + 3 + step &&
            actual[1].logicalLength() == reference[0].logicalLength() &&
            actual[1].logicalLength() == independent[0].logicalLength() && trunk.ownsState(actual[1]),
            "cancel survivor continuation metadata differs");
    if (step) record << ',';
    record << "{\"step\":" << step << ",\"incoming_token\":" << token << ",\"scalar_verify_commit_reference\":"; compared.write(record);
    record << ",\"ordinary_retained_prefix_reference\":"; prefixContinuation.write(record); record << '}';
  }
  record << "],\"pass\":" << (pass ? "true" : "false") << '}';
  report.cases.push_back(record.str()); report.pass &= pass;
  {
    auto cancelled = fresh(trunk, 2); auto live = pointers(cancelled);
    const std::array<uint32_t, 2> incoming{tokens.front(), tokens.front()}, zero{0, 0};
    const auto result = batch.verifyBatch(live, incoming, 1);
    (void)snapshot(result, 2, 1); report.timings.add("all_cancel_verify", result.timing);
    const std::array<FlashRequestState *, 2> nulls{nullptr, nullptr};
    const auto timing = batch.commitBatch(nulls, zero); report.timings.add("all_cancel_commit", timing, false);
    require(!batch.pending() && cancelled[0].poisoned() && cancelled[1].poisoned() &&
            timing.gpuSeconds == 0 && timing.wallSeconds == 0, "all-terminal commit did not cancel both lanes without a command");
    records(report, "all_zero_retention_with_null_slots_is_terminal_and_submits_no_restore");
  }
  std::cerr << "flash-batch-verify-oracle mixed_cancel pass=" << (pass ? "true" : "false") << '\n';
}
void cpuSelfTest() {
  Error exact, mismatch, nonfinite;
  const std::vector<uint16_t> reference{0x3f80, 0x4000, 0xbf80, 0x0000};
  exact.add(reference, reference, 3, 2); auto changed = reference; changed[1] = 0x4001;
  mismatch.add(changed, reference, 2, 2); changed[0] = 0x7fc0; nonfinite.add(changed, reference, 1, 2);
  require(exact.pass() && exact.elements == 4 && !mismatch.pass() && mismatch.mismatch == 1 &&
          mismatch.maximumULP == 1 && mismatch.first[0].lane == 2 && mismatch.first[0].row == 0 &&
          mismatch.first[0].column == 1 && nonfinite.nonfinite == 1 && !nonfinite.pass() &&
          greedyOrInvalid(changed) == std::numeric_limits<uint32_t>::max() &&
          continuationAnchor(changed) == 0, "comparison CPU self-test failed");
  const std::vector<uint16_t> tied{0x3f80, 0x4000, 0x4000, 0xbf80};
  require(greedy(tied) == 1 && sha256(reference).size() == 64 && choices("4,2,4") == std::vector<uint32_t>{2, 4},
          "greedy/hash/selector CPU self-test failed");
  std::ostringstream json; nonfinite.write(json);
  require(json.str().find("null") != std::string::npos && json.str().find("nan") == std::string::npos,
          "nonfinite diagnostic JSON CPU self-test failed");
  std::cout << "{\"pass\":true,\"mode\":\"cpu-self-test\",\"observed_gpu_commands\":0}\n";
}
void help() {
  std::cout << "usage: flash-batch-verify-oracle METALLIB PACKAGE TOKENS_JSON REPORT_JSON\n"
               "       flash-batch-verify-oracle --cpu-self-test\n"
               "TOKENS_JSON: 1..64 integer tokens; lane prompts cycle/rotate these to 8+ rows.\n"
               "Default: all B1..B4 x rows1..4, two verify/commit trials, five forced AR steps.\n"
               "B4/rows4 and mixed B2 cancellation also compare ordinary retained-prefix states.\n"
               "Optional FLASH_BATCH_VERIFY_LANES and FLASH_BATCH_VERIFY_ROWS: comma lists 1..4.\n"
               "Optional FLASH_BATCH_VERIFY_SOURCE_ROOT: repository root for real source hashes.\n"
               "Short contexts do not qualify QSA sparse top512 block selection.\n";
}
} // namespace
int main(int argc, char **argv) {
  @autoreleasepool {
    Report report;
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") { help(); return 0; }
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (argc != 5) { help(); return 2; }
      report.root = sourceRoot(); report.sources = sourceHashes(report.root);
      const auto packageRoot = std::filesystem::canonical(argv[2]);
      report.inputPaths.emplace("executable", executablePath());
      report.inputPaths.emplace("metallib", std::filesystem::canonical(argv[1]));
      report.inputPaths.emplace("package_manifest", packageRoot / "manifest.json");
      report.inputPaths.emplace("tokens_json", std::filesystem::canonical(argv[3]));
      for (const auto &[name, path] : report.inputPaths) report.inputs.emplace(name, fileSHA256(path));
      report.lanes = choices(std::getenv("FLASH_BATCH_VERIFY_LANES"));
      report.rows = choices(std::getenv("FLASH_BATCH_VERIFY_ROWS"));
      const auto tokenPath = report.inputPaths.at("tokens_json").string();
      const auto tokens = loadTokens(tokenPath.c_str());
      CanonicalEnvironment canonical;
      MetalBackend backend(report.inputPaths.at("metallib").string());
      backend.setCommandDispatchProfiling(splash::metal::CommandDispatchProfilingMode::Off);
      const auto weights = FlashWeights::load(backend, packageRoot);
      require(weights.descriptor().vocabularySize == kVocabulary && weights.descriptor().hiddenSize == 2560 &&
              weights.descriptor().layers == 48, "oracle requires the native Flash-Next 48-layer text trunk");
      report.sourceIdentity = weights.sourceIdentity(); report.manifestFingerprint = weights.manifestFingerprint();
      FlashForward trunk(backend, weights, kCapacity, 128, 4); FlashBatchVerify batch(backend, weights, trunk, kCapacity, 4, 4);
      report.kernelRoutes = trunk.kernelRoutes(); report.trunkBytes = trunk.workspaceBytes(); report.batchBytes = batch.workspaceBytes();
      for (const auto lanes : report.lanes)
        for (const auto rows : report.rows) runCase(trunk, batch, lanes, rows, tokens, report);
      runSafety(backend, weights, trunk, batch, tokens.front(), report);
      runCancellation(trunk, batch, tokens, report); require(!batch.pending(), "oracle left a pending trial");
      report.complete = true; report.sourceStable = sourceHashes(report.root) == report.sources;
      report.inputsStable = true;
      for (const auto &[name, path] : report.inputPaths)
        report.inputsStable &= fileSHA256(path) == report.inputs.at(name);
      require(report.sourceStable, "source files changed during the oracle run");
      require(report.inputsStable, "executable/metallib/manifest/prompt changed during the oracle run");
      report.write(argv[4]); return report.pass ? 0 : 1;
    } catch (const std::exception &error) {
      report.pass = false; report.failure = error.what();
      if (argc == 5) {
        try { report.write(argv[4]); }
        catch (const std::exception &writeError) { std::cerr << "report: " << writeError.what() << '\n'; }
      }
      std::cerr << "flash-batch-verify-oracle: " << error.what() << '\n'; return 1;
    }
  }
}
