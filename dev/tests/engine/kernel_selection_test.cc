// The kernel selection golden: every plan the device policy picks for the
// projections, float tiles, MoE blocks and attention of the families Splash
// serves, on each device class and core count, over their formats and loads.
// Each (device, section, group) holds the FNV-1a-64 of its plan lines; the
// golden file holds one line per group. A change of selection shows as the
// groups it changes, and --dump <dir> writes every plan line to diff them.
//
//   kernel-selection <golden>            compare against the golden
//   kernel-selection --write <golden>    write it
//   kernel-selection --dump <dir> [...]  also write the plan lines
#include "TestChecks.hpp"
#include "model/ModelDescriptor.hpp"
#include "ops/ExecutionPlans.hpp"

#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

// The devices: Apple9 and Apple10 at the core counts the plans tell apart and
// the CI runners' Apple9 sizes. Family 11 plans as Apple10 (gpuClass), so its
// 12 and 20 cores stand for Apple10's. Zero cores is a device that reports
// none, planned at kAssumedGpuCores (32) on every family, so Apple9's alone.
constexpr std::pair<uint32_t, uint32_t> kDevices[] = {
    {9, 0},   {9, 1},   {9, 10},  {9, 11},  {9, 14},  {9, 16},  {9, 20},  {9, 40},
    {10, 16}, {10, 32}, {10, 40}, {11, 12}, {11, 20}};

DeviceCapabilities device(uint32_t family, uint32_t cores) {
  DeviceCapabilities result;
  result.appleGpuFamily = family;
  result.gpuCoreCount = cores;
  return result;
}

// One group's plan lines, hashed as they come.
class Group final {
public:
  void add(const std::string &line) {
    for (const char c : line) hash_ = (hash_ ^ uint8_t(c)) * 1099511628211ULL;
    hash_ = (hash_ ^ uint8_t('\n')) * 1099511628211ULL;
    if (dump_) lines_.push_back(line);
  }
  void keepLines() { dump_ = true; }
  [[nodiscard]] uint64_t hash() const noexcept { return hash_; }
  [[nodiscard]] const std::vector<std::string> &lines() const noexcept { return lines_; }

private:
  uint64_t hash_ = 14695981039346656037ULL;
  bool dump_ = false;
  std::vector<std::string> lines_;
};

// Every group of every device, keyed "<family>,<cores> <section> <group>".
class Golden final {
public:
  explicit Golden(bool dump) : dump_(dump) {}
  Group &group(const std::string &device, const std::string &section, const std::string &name) {
    const std::string key = device + " " + section + " " + name;
    auto [it, inserted] = groups_.try_emplace(key);
    if (inserted && dump_) it->second.keepLines();
    return it->second;
  }
  [[nodiscard]] const std::map<std::string, Group> &groups() const noexcept { return groups_; }

private:
  bool dump_;
  std::map<std::string, Group> groups_;
};

// The line of a call that may throw: its result, or what it threw.
template <class F> std::string attempt(F describe) {
  try {
    return describe();
  } catch (const std::exception &error) {
    return std::string("throws ") + error.what();
  }
}

std::string formatName(uint32_t format) {
  if (format == QuantizedSegment::kFloat32) return "f32";
  return format < GGUF_FMT_COUNT ? kQuantFormats[format].name : "none";
}

// ---- Projections ----------------------------------------------------------

struct ProjectionCase final {
  std::string name;
  uint32_t n, k, segments;
};

// The projections of the anchors in linear_plan_test.mm and of each family:
// its GDN and attention inputs (whole and as fused segments), their outputs,
// the dense FFN's and the shared expert's gate/up and down, the head and the
// hyper-connection mixes. A matrix the plans refuse is a case too.
std::vector<ProjectionCase> projectionCases() {
  std::vector<ProjectionCase> cases{
      {"27b-down", 5120, 17408, 1},          {"35b-output", 2048, 4096, 1},
      {"35b-shared-gate-up", 512, 2048, 1},  {"27b-gate-up", 17408, 5120, 1},
      {"fused-two", 4096, 5120, 2},          {"27b-gdn-input", 16640, 5120, 3},
      {"27b-head", 248320, 5120, 1},         {"narrow-1024x256", 1024, 256, 1},
      {"narrow-1024x3072", 1024, 3072, 1},   {"27b-out", 5120, 6144, 1},
      {"27b-attention-input", 14336, 5120, 3}, {"35b-gdn-input", 12288, 2048, 2},
      {"35b-shared-down", 2048, 512, 1},     {"35b-head", 248320, 2048, 1},
      {"narrow-1024x5120", 1024, 5120, 1},   {"narrow-512x5120", 512, 5120, 1},
      {"narrow-1024x1024", 1024, 1024, 1},   {"flash-next-hyper-down", 320, 10240, 1},
  };
  model::forEachFamily([&](const auto &family) {
    const std::string f(family.family);
    const uint32_t h = family.hiddenSize;
    const auto add = [&](const char *name, uint32_t n, uint32_t k, uint32_t segments) {
      if (!n || !k) return;
      for (const ProjectionCase &c : cases)
        if (c.n == n && c.k == k && c.segments == segments) return;
      cases.push_back({f + "-" + name, n, k, segments});
    };
    if (family.gdnLayerCount()) {
      add("gdn-input", family.packedGdnWidth, h, 1);
      add("gdn-input-fused", family.packedGdnWidth, h, 3);
      add("gdn-output", h, family.gdnValueHeads * family.gdnHeadDimension, 1);
    }
    add("attention-input", family.packedFullWidth, h, 1);
    add("attention-input-fused", family.packedFullWidth, h, 3);
    add("attention-output", h, family.attentionWidth, 1);
    add("ffn-gate-up", family.intermediateSize, h, 1);
    add("ffn-down", h, family.intermediateSize, 1);
    add("shared-gate-up", family.sharedExpertSlices * family.expertIntermediateSize, h, 1);
    add("shared-down", h, family.sharedExpertSlices * family.expertIntermediateSize, 1);
    add("head", family.vocabularySize, h, 1);
    if (family.hyperConnected()) {
      add("hyper-down", family.hyperRank, family.streamWidth(), 1);
      add("hyper-up", family.streamWidth(), family.hyperRank, 1);
    }
  });
  return cases;
}

// The formats of a projection's segments, in turn: each GGUF format alone and
// mixed units as the matrix GGUFs hold them (a UD quant's grid-lookup pair, a
// K-quant pair, Q2_K beside IQ3_XXS, a quantized unit with float segments).
struct FormatSet final {
  std::string name;
  std::vector<uint32_t> formats;
};
std::vector<FormatSet> formatSets() {
  std::vector<FormatSet> sets;
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format) sets.push_back({formatName(format), {format}});
  sets.push_back({"iq2_xs+iq3_xxs", {GGUF_FMT_IQ2XS, GGUF_FMT_IQ3XXS}});
  sets.push_back({"q4_k+q6_k", {GGUF_FMT_Q4K, GGUF_FMT_Q6K}});
  sets.push_back({"q2_k+iq3_xxs", {GGUF_FMT_Q2K, GGUF_FMT_IQ3XXS}});
  sets.push_back({"q4_k+f32", {GGUF_FMT_Q4K, QuantizedSegment::kFloat32}});
  return sets;
}

QuantizedSegment segment(uint32_t format, uint32_t n, uint32_t k) {
  return format == QuantizedSegment::kFloat32 ? QuantizedSegment::floats(n, k, {})
                                              : QuantizedSegment::planes(format, n, k, {}, {}, {});
}

// A projection of `c.segments` equal segments in the set's formats from the
// `first`-th on; planning reads only their geometry and formats.
Projection projection(const ProjectionCase &c, const FormatSet &set, size_t first = 0) {
  BlockWeights weights;
  const uint32_t width = c.n / c.segments;
  for (uint32_t s = 0; s < c.segments; ++s) {
    const uint32_t columns = s + 1 == c.segments ? c.n - width * s : width;
    QuantizedSegment q = segment(set.formats[(first + s) % set.formats.size()], columns, c.k);
    q.columnOffset = width * s;
    weights.segments.push_back(q);
  }
  return Projection(c.n, c.k, std::move(weights));
}

std::string describe(const LinearPlan &plan) {
  const LinearConfig c = plan.configuration();
  const LinearScratchSize s = plan.scratchSize();
  std::ostringstream line;
  line << "tile " << unsigned(c.tile) << " splits " << c.splits << " spread " << c.spread << " one-pass "
       << c.oneGateUpPass << " sixteen " << c.sixteenRows << " storage " << plan.storageRows() << " input "
       << unsigned(plan.input()) << " scratch " << s.input << "/" << s.sums << "/" << s.partials << "/"
       << s.counters << "/" << s.rotated << " gate " << plan.gateScratchBytes();
  return line.str();
}

std::string describe(const LinearScratchSize &s) {
  std::ostringstream line;
  line << s.input << "/" << s.sums << "/" << s.partials << "/" << s.counters << "/" << s.rotated;
  return line.str();
}

constexpr uint32_t kPrefillRows[] = {1, 8, 9, 16, 17, 24, 32, 33, 48, 64, 65, 128, 512, 2048};

void projections(Golden &golden, const std::string &name, const ExecutionPlans &plans) {
  const Linear &linear = plans.linear();
  const std::vector<FormatSet> sets = formatSets();
  for (const ProjectionCase &c : projectionCases()) {
    Group &decode = golden.group(name, "projection-decode", c.name);
    Group &prefill = golden.group(name, "projection-prefill", c.name);
    Group &shapes = golden.group(name, "projection-shape", c.name);
    for (const FormatSet &set : sets) {
      const std::optional<Projection> p = [&]() -> std::optional<Projection> {
        try {
          return projection(c, set);
        } catch (const std::exception &) {
          return std::nullopt;
        }
      }();
      if (!p) continue;
      const Projection gate = projection(c, set, 1);
      for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
          const LinearWorkload w{{c.n, c.k}, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
          decode.add(set.name + " lanes " + std::to_string(lanes) + " e" + std::to_string(unsigned(epilogue)) +
                     ": " + attempt([&] {
                       return describe(linear.plan(w, *p, epilogue == LinearEpilogue::GateUp ? &gate : nullptr));
                     }));
        }
      for (const uint32_t rows : kPrefillRows)
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate}) {
          const LinearWorkload w{{c.n, c.k}, rows, LinearPhase::Prefill, epilogue};
          prefill.add(set.name + " rows " + std::to_string(rows) + " e" + std::to_string(unsigned(epilogue)) +
                      ": " + attempt([&] { return describe(linear.plan(w, *p)); }));
        }
      // The arena bounds, which read the shape alone, and its rotated twin.
      for (const bool rotated : {false, true}) {
        ProjectionShape shape = p->shape();
        shape.rotated = rotated;
        std::string line = set.name + (rotated ? " rotated" : "") + ":";
        for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
          line += " storage " + attempt([&] {
            return std::to_string(linear.decodeStorageRows(lanes * SPLASH_TARGET_VERIFY_ROWS, shape));
          });
        line += " decode " + attempt([&] { return describe(linear.decodeScratchSize(shape)); });
        line += " prefill " + attempt([&] { return describe(linear.prefillScratchSize(shape)); });
        line += " gate-up " + attempt([&] { return std::to_string(plans.gateUpWorkspace(shape)); });
        shapes.add(line);
      }
    }
    // Plans of the matrix without its projections' formats.
    Group &matrix = golden.group(name, "projection-matrix", c.name);
    for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
        matrix.add("lanes " + std::to_string(lanes) + " e" + std::to_string(unsigned(epilogue)) + ": " +
                   attempt([&] {
                     return describe(linear.plan(
                         {{c.n, c.k}, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue}));
                   }));
    for (const uint32_t rows : kPrefillRows)
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
        matrix.add("rows " + std::to_string(rows) + " e" + std::to_string(unsigned(epilogue)) + ": " +
                   attempt([&] {
                     return describe(linear.plan({{c.n, c.k}, rows, LinearPhase::Prefill, epilogue}));
                   }));
  }
}

// The float tile of rows x columns of each float format and quantized rows.
void floatTiles(Golden &golden, const std::string &name, const Linear &linear) {
  constexpr uint32_t formats[] = {QuantizedSegment::kFloat32, GGUF_FMT_BF16, GGUF_FMT_Q4K, GGUF_FMT_Q80,
                                  GGUF_FMT_AF8G64, GGUF_FMT_AF4G64};
  for (const uint32_t format : formats) {
    Group &group = golden.group(name, "float-tile", formatName(format));
    for (const uint32_t columns : {8U, 64U, 96U, 128U, 256U, 320U, 512U, 1024U, 2048U, 4096U}) {
      std::string line = "columns " + std::to_string(columns) + ":";
      for (uint32_t rows = 1; rows <= 2048; rows += rows < 64 ? 1 : 16)
        line += linear.ggufFloatTile(rows, columns, format) == FloatTile::NeuralAccelerator ? 'a' : 's';
      group.add(line);
    }
  }
}

// ---- MoE ------------------------------------------------------------------

// The formats of a target's MoE blocks: each GGUF format in every block, and
// the mixes the matrix GGUFs and their failure cases hold.
struct BlockFormats final {
  uint32_t gate, up, down, shared, router;
};
struct MoeFormatSet final {
  std::string name;
  std::vector<BlockFormats> blocks;
};
std::vector<MoeFormatSet> moeFormatSets() {
  constexpr uint32_t f32 = QuantizedSegment::kFloat32;
  std::vector<MoeFormatSet> sets;
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format)
    sets.push_back({formatName(format), std::vector<BlockFormats>(4, {format, format, format, format, f32})});
  // A UD quant's experts: grid-lookup gate/up over an IQ3_XXS down, Q8_0 shared.
  sets.push_back({"ud-iq2", std::vector<BlockFormats>(4, {GGUF_FMT_IQ2XS, GGUF_FMT_IQ2XS, GGUF_FMT_IQ3XXS,
                                                          GGUF_FMT_Q80, f32})});
  // Grid-lookup experts in every block but the last.
  sets.push_back({"iq2-then-q4_k", {{GGUF_FMT_IQ2XS, GGUF_FMT_IQ2XS, GGUF_FMT_IQ2XS, GGUF_FMT_Q80, f32},
                                    {GGUF_FMT_IQ2XS, GGUF_FMT_IQ2XS, GGUF_FMT_IQ2XS, GGUF_FMT_Q80, f32},
                                    {GGUF_FMT_IQ2XS, GGUF_FMT_IQ2XS, GGUF_FMT_IQ2XS, GGUF_FMT_Q80, f32},
                                    {GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q80, f32}}});
  // A shared expert the register tile does not take.
  sets.push_back({"q4_k-shared-f16", std::vector<BlockFormats>(4, {GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K,
                                                                   GGUF_FMT_F16, f32})});
  // Routers in BF16, in quantized rows, and of mixed formats.
  sets.push_back({"router-bf16", std::vector<BlockFormats>(4, {GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K,
                                                               GGUF_FMT_Q4K, GGUF_FMT_BF16})});
  sets.push_back({"router-q8_0", std::vector<BlockFormats>(4, {GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K,
                                                               GGUF_FMT_Q4K, GGUF_FMT_Q80})});
  sets.push_back({"router-mixed", {{GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, f32},
                                   {GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_BF16},
                                   {GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, GGUF_FMT_BF16}}});
  return sets;
}

std::vector<BlockMoeWeights> moeBlocks(const MoeShape &shape, const MoeFormatSet &set) {
  const uint32_t h = shape.hiddenSize, n = shape.expertIntermediateSize, slots = shape.experts;
  const uint32_t sharedN = n * shape.sharedSlices;
  std::vector<BlockMoeWeights> blocks;
  for (const BlockFormats &f : set.blocks) {
    BlockMoeWeights block;
    block.router = QuantizedSegment::rows(f.router, shape.experts, h, {});
    block.gate = {segment(f.gate, slots * n, h), segment(f.shared, sharedN, h)};
    block.up = {segment(f.up, slots * n, h), segment(f.shared, sharedN, h)};
    block.down = {segment(f.down, slots * h, n), segment(f.shared, h, sharedN)};
    blocks.push_back(std::move(block));
  }
  return blocks;
}

// The formats of the target's MoE blocks, as its loaded weights give them
// (QwenTargetGeometry::finish).
MoeFormats formatsOf(const std::vector<BlockMoeWeights> &blocks) {
  std::vector<const BlockMoeWeights *> pointers;
  for (const BlockMoeWeights &block : blocks) pointers.push_back(&block);
  return moeFormats(pointers);
}

std::string describe(const MoePlan &plan) {
  const MoeConfig c = plan.configuration();
  std::ostringstream line;
  line << "tile " << unsigned(c.expertTile) << " gguf " << unsigned(c.ggufTile) << " router "
       << unsigned(c.ggufRouterTile) << " tiles " << plan.maximumTiles() << " workspace";
  for (const auto field : kMoeWorkspaceFields) line << " " << plan.workspace().*field;
  return line.str();
}
std::string describe(const MoeWorkspace &workspace) {
  std::string line;
  for (const auto field : kMoeWorkspaceFields) line += " " + std::to_string(workspace.*field);
  return line;
}

void moePlans(Golden &golden, const std::string &name, const ExecutionPlans &plans) {
  const std::vector<MoeFormatSet> sets = moeFormatSets();
  model::forEachFamily([&](const auto &family) {
    if (family.ffnKind != model::QwenFfnKind::SparseMoe) return;
    const std::string f(family.family);
    for (const MoeFormatSet &set : sets) {
      const MoeShape shape = family.moeBlockShape();
      const MoeFormats formats = formatsOf(moeBlocks(shape, set));
      Group &decode = golden.group(name, "moe-decode", f);
      Group &prefill = golden.group(name, "moe-prefill", f);
      Group &workspace = golden.group(name, "moe-workspace", f);
      for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
        decode.add(set.name + " lanes " + std::to_string(lanes) + ": " +
                   attempt([&] { return describe(plans.moeDecode(shape, formats, lanes)); }));
      for (uint32_t rows = 1; rows <= SPLASH_PREFILL_TOKEN_BUDGET; rows += rows < 128 ? 1 : 16)
        prefill.add(set.name + " rows " + std::to_string(rows) + ": " +
                    attempt([&] { return describe(plans.moePrefill(shape, formats, rows)); }));
      for (const uint32_t rows : {64U, 512U, uint32_t{SPLASH_PREFILL_TOKEN_BUDGET}})
        workspace.add(set.name + " prefill " + std::to_string(rows) + ":" +
                      attempt([&] { return describe(plans.moePrefillWorkspace(shape, formats, rows)); }));
      workspace.add(set.name + " decode per lane:" +
                    attempt([&] { return describe(plans.moeDecodeWorkspacePerLane(shape, formats)); }));
    }
  });
}

// ---- Attention ------------------------------------------------------------

std::string describe(const metal::DispatchSize &size) {
  return std::to_string(size.x) + "x" + std::to_string(size.y) + "x" + std::to_string(size.z);
}
std::string describe(const AttentionWorkspace &workspace) {
  return std::to_string(workspace.partialsBytes) + "/" + std::to_string(workspace.statisticsBytes);
}
std::string describe(const PrefillAttentionPlan &plan) {
  return plan.splitPipeline + " " + plan.reducePipeline + " tiles " + std::to_string(plan.tiles) + " splits " +
         std::to_string(plan.splits) + " groups " + describe(plan.splitGroups) + " " + describe(plan.reduceGroups) +
         " workspace " + describe(plan.workspace) + " sparse " + std::to_string(plan.sparse) + " sinks " +
         std::to_string(plan.sinks);
}
std::string describe(const VerifyAttentionPlan &plan) {
  std::string splits;
  for (const uint32_t s : plan.laneSplits) splits += std::to_string(s) + ",";
  return plan.splitPipeline + " " + plan.reducePipeline + " splits " + splits + std::to_string(plan.splits) +
         " groups " + describe(plan.splitGroups) + " " + describe(plan.reduceGroups) + " workspace " +
         describe(plan.workspace) + " sparse " + std::to_string(plan.sparse) + " sinks " +
         std::to_string(plan.sinks);
}

void attentionPlans(Golden &golden, const std::string &name, const ExecutionPlans &plans) {
  constexpr uint32_t rowsList[] = {1, 7, 8, 64, 255, 256, 300, 512, 1024, 2048};
  constexpr uint32_t histories[] = {0, 1, 511, 2047, 4096, 32768, 65536, 131072, 262144};
  model::forEachFamily([&](const auto &family) {
    const std::string f(family.family);
    // The variants its attention layers run (QwenTargetGeometry::attentionVariant).
    std::vector<AttentionVariant> variants;
    const kv::Layout base = family.kvLayout();
    for (uint32_t a = 0; a < base.attentionLayers; ++a) {
      const AttentionVariant v{.rotated = a >= 64 || !(family.unrotatedLayers >> a & 1),
                               .windowTokens = base.windowed(a) ? base.windowTokens : 0};
      if (std::find(variants.begin(), variants.end(), v) == variants.end()) variants.push_back(v);
    }
    Group &prefill = golden.group(name, "attention-prefill", f);
    Group &verify = golden.group(name, "attention-verify", f);
    Group &workspace = golden.group(name, "attention-workspace", f);
    for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
      kv::Layout layout = base;
      layout.format = format;
      const std::string fmt = format == kv::Format::Int8 ? "int8" : "bf16";
      for (const AttentionVariant &v : variants) {
        const std::string variant =
            fmt + (v.rotated ? " rope" : " nope") + " window " + std::to_string(v.windowTokens);
        for (const uint32_t rows : rowsList)
          for (const uint32_t history : histories)
            prefill.add(variant + " rows " + std::to_string(rows) + " history " + std::to_string(history) + ": " +
                        attempt([&] {
                          return describe(
                              plans.prefillAttention(rows, history, family.attentionQueryHeads, layout, v));
                        }));
        for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
          for (const uint32_t history : histories) {
            std::vector<uint32_t> each(lanes, history);
            if (lanes > 1) each.front() = history / 2;
            verify.add(variant + " lanes " + std::to_string(lanes) + " history " + std::to_string(history) + ": " +
                       attempt([&] {
                         return describe(plans.verifyAttention(lanes, family.attentionQueryHeads, layout, each, v));
                       }));
          }
      }
      for (const uint32_t rows : {1U, 256U, uint32_t{SPLASH_PREFILL_TOKEN_BUDGET}})
        workspace.add(fmt + " prefill " + std::to_string(rows) + ": " + attempt([&] {
                        return describe(plans.prefillAttentionWorkspace(rows, family.attentionQueryHeads, layout));
                      }));
      workspace.add(fmt + " verify per lane: " + attempt([&] {
                      return describe(plans.verifyAttentionWorkspacePerLane(family.attentionQueryHeads, layout));
                    }));
    }
  });
}

// ---- Golden file ----------------------------------------------------------

Golden selections(bool dump) {
  Golden golden(dump);
  for (const auto [family, cores] : kDevices) {
    const std::string name = std::to_string(family) + "," + std::to_string(cores);
    const ExecutionPlans plans(device(family, cores));
    projections(golden, name, plans);
    floatTiles(golden, name, plans.linear());
    moePlans(golden, name, plans);
    attentionPlans(golden, name, plans);
  }
  return golden;
}

std::string hex(uint64_t value) {
  char text[17];
  std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
  return text;
}

std::map<std::string, std::string> read(const std::string &path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::map<std::string, std::string> entries;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    const size_t space = line.rfind(' ');
    entries[line.substr(0, space)] = line.substr(space + 1);
  }
  return entries;
}

void dumpLines(const Golden &golden, const std::string &directory) {
  std::filesystem::create_directories(directory);
  std::map<std::string, std::ofstream> files;
  for (const auto &[key, group] : golden.groups()) {
    std::istringstream parts(key);
    std::string device, section, name;
    parts >> device >> section >> name;
    std::string file = device + "_" + section + ".txt";
    std::replace(file.begin(), file.end(), ',', '_');
    auto [it, inserted] = files.try_emplace(file);
    if (inserted) it->second.open(std::filesystem::path(directory) / file);
    for (const std::string &line : group.lines()) it->second << name << " | " << line << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::optional<std::string> dump, golden, write;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--dump" && i + 1 < argc) dump = argv[++i];
      else if (arg == "--write" && i + 1 < argc) write = argv[++i];
      else golden = arg;
    }
    test::require(golden.has_value() != write.has_value(), "usage: kernel-selection [--dump <dir>] "
                                                           "(<golden> | --write <golden>)");
    const Golden computed = selections(dump.has_value());
    if (dump) dumpLines(computed, *dump);
    if (write) {
      std::ofstream file(*write);
      file << "# kernel_selection_test.cc: FNV-1a-64 of each group's plan lines\n";
      for (const auto &[key, group] : computed.groups()) file << key << ' ' << hex(group.hash()) << '\n';
      std::cout << "wrote " << computed.groups().size() << " groups to " << *write << '\n';
      return 0;
    }
    const std::map<std::string, std::string> expected = read(*golden);
    std::vector<std::string> differ;
    for (const auto &[key, group] : computed.groups()) {
      const auto it = expected.find(key);
      if (it == expected.end()) differ.push_back("new " + key);
      else if (it->second != hex(group.hash())) differ.push_back("changed " + key);
    }
    for (const auto &[key, hash] : expected)
      if (!computed.groups().contains(key)) differ.push_back("gone " + key);
    if (!differ.empty()) {
      for (size_t i = 0; i < differ.size() && i < 40; ++i) std::cerr << differ[i] << '\n';
      throw std::runtime_error(std::to_string(differ.size()) + " of " + std::to_string(computed.groups().size()) +
                               " selection groups differ from the golden");
    }
    std::cout << "PASS kernel selection: " << computed.groups().size() << " groups of " << std::size(kDevices)
              << " devices match the golden\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL kernel selection: " << error.what() << '\n';
    return 1;
  }
}
