#pragma once

// Small models whose weights load, which the runtime plans and starts as it
// would a real installation's, and the volumes their SSD caches find.

#include "GgufFixtures.hpp"
#include "TestCheckpoint.hpp"
#include "TestFiles.hpp"
#include "TestGguf.hpp"
#include "engine/DiskSpace.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/DFlashDraft.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen4Exp.hpp"
#include "ops/Vision.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace splash::test {

// A small model whose weights load and whose geometry the runtime plans:
// four target and two draft layers of the 2048-wide hidden state a compiled
// draft attention takes, the draft reading a GDN layer and the attention
// layer, with Qwen3.8-27B's attention and GDN heads and a two-block vision
// tower.
struct PlannedModel final {
  model::Qwen3_8Layout target;
  model::DFlashDraftLayout draft;
  ops::VisionLayout vision;
};

inline PlannedModel plannedModel() {
  PlannedModel result;
  model::Qwen3_8Layout &target = result.target;
  target.layers = 4;
  target.hiddenSize = 2048;
  target.vocabularySize = 1024;
  target.stopTokens = {1022, 1023};
  target.intermediateSize = 1024;
  constexpr std::array<uint32_t, 2> draftReads{1, 3};
  result.draft = model::draftLayout(target.hiddenSize, target.vocabularySize, 2, 1024, 10'000'000.0F, draftReads);
  ops::VisionLayout &vision = result.vision;
  vision.depth = 2;
  vision.hiddenSize = 128;
  vision.patchDimension = 1536;
  vision.intermediateSize = 200;
  vision.paddedIntermediateSize = 256;
  vision.mergedHiddenSize = 512;
  vision.outputHiddenSize = target.hiddenSize;
  vision.heads = 2;
  vision.headDimension = 64;
  vision.positionGridSide = 4;
  return result;
}

// A model under root whose weights load, which nothing but the host
// refuses: `name` names its descriptor and build.
inline engine::RuntimeResourcesConfig loadableConfig(const char *metallibPath, const std::filesystem::path &root,
                                                     const std::string &name) {
  const PlannedModel planned = plannedModel();
  static_cast<void>(test::writeSyntheticModel(root, planned.target, planned.draft, planned.vision));
  engine::RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor(name, planned.target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  config.model.sourceIdentity = "synthetic";
  config.buildId = name;
  config.hostMemory = [] {
    return std::optional<engine::HostMemory>(engine::HostMemory{64 * engine::kGiB, 64 * engine::kGiB});
  };
  // The tests of this model check the start's memory and context rules, not
  // the Neural Engine split, which takes its 2048-wide FFN: it stays off. The
  // split fails at every start here (its check against the GPU alone is 0/0
  // on these zero weights), so a start would serve the GPU alone at the
  // split's smaller context, after freeing the prefill arena its setup
  // borrowed just before the KV runway.
  config.aneFfn.enabled = false;
  return config;
}

// A local volume of 1000 GiB with `availableBytes` free, which leaves the
// automatic SSD cache room from 12 GiB on.
inline engine::DiskVolume localVolume(uint64_t availableBytes) {
  return {"/Volumes/Test", availableBytes, 1000 * engine::kGiB, true};
}

// A volume probe that reads `volume` for every directory.
inline std::function<std::optional<engine::DiskVolume>(const std::filesystem::path &)>
reading(std::optional<engine::DiskVolume> volume) {
  return [volume](const std::filesystem::path &) { return volume; };
}

// The routed experts of each layer of the MoE model below.
inline constexpr uint32_t kMoeExperts = 128;
inline constexpr uint32_t kMoeLayers = 8;

// A MoE model under root whose weights load: the planned model's draft and
// vision tower beside kMoeLayers layers of Qwen3.6-35B-A3B's geometry, each
// with kMoeExperts routed experts, eight a token. Its experts above their
// smallest store, a layer's, take more than the arenas of its prefill groups
// and the first 64K tokens of its KV.
inline engine::RuntimeResourcesConfig moeConfig(const char *metallibPath, const std::filesystem::path &root) {
  const PlannedModel planned = plannedModel();
  model::Qwen3_6MoeLayout target;
  target.layers = kMoeLayers;
  target.vocabularySize = planned.target.vocabularySize;
  target.stopTokens = planned.target.stopTokens;
  target.experts = kMoeExperts;
  static_cast<void>(test::writeSyntheticModel(root, target, planned.draft, planned.vision));
  engine::RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("moe-test", target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "moe-test";
  config.hostMemory = [] {
    return std::optional<engine::HostMemory>(engine::HostMemory{64 * engine::kGiB, 64 * engine::kGiB});
  };
  return config;
}

// A Qwen3.8-Flash-Next model under root whose weights load, a block-sparse
// target whose KV spills: a GGUF of its family's tensors
// (gguf_fixtures::targetTensors) with its attention, GDN, indexer and PLE
// heads but a hidden state of 256 values, a vocabulary of 256 tokens, eight
// layers, every second one full attention, and four routed experts a layer,
// a PLE table of `pleRows` rows, and a draft of one layer that reads four of
// them; no vision tower.
inline engine::RuntimeResourcesConfig flashNextConfig(const char *metallibPath, const std::filesystem::path &root,
                                                      uint64_t pleRows = gguf_fixtures::kPleTableRows) {
  model::Qwen4ExpLayout target;
  target.layers = 8;
  target.hiddenSize = 256;
  target.vocabularySize = 256;
  target.fullAttentionPeriod = 2;
  target.maskToken = 252;
  target.stopTokens = {250, 251};
  target.pleEosToken = 250;
  target.pleImageToken = 253;
  target.experts = 4;
  target.expertsPerToken = 2;
  target.expertIntermediateSize = 256;
  target.pleLayer = 0;
  gguf_fixtures::TensorTypes types = gguf_fixtures::qwen4ExpTypes();
  for (const char *name : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
                           "output_hc_down.weight", "output_hc_up.weight", "ffn_gate_exps.weight",
                           "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                           "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
    types[name] = gguf_fixtures::kQ8_0;
  // The PLE hash, its heads an equal share of the table's rows each.
  std::vector<test::gguf::Key> keys = gguf_fixtures::metadata(target);
  const std::string ple = std::string(model::Qwen4ExpLayout::ggufArchitecture) + ".ple.";
  const uint32_t heads = target.pleHeads(), headRows = static_cast<uint32_t>(pleRows / heads);
  std::vector<int32_t> offsets, rows(heads, int32_t(headRows));
  for (uint32_t head = 0; head < heads; ++head)
    offsets.push_back(int32_t(head * headRows));
  keys.push_back(test::gguf::int32ArrayKey(ple + "layer_multipliers", {1, 31, 961}));
  keys.push_back(test::gguf::int32ArrayKey(ple + "head_offsets", offsets));
  keys.push_back(test::gguf::int32ArrayKey(ple + "head_vocab_sizes", rows));
  keys.push_back(test::gguf::uint32Key(ple + "eos_token_id", target.pleEosToken));
  std::filesystem::create_directories(root / "target");
  test::writeFile(root / "target" / "model.gguf",
                  test::gguf::file(keys, gguf_fixtures::targetTensors(target, types, pleRows)));
  constexpr std::array<uint32_t, 4> draftReads{1, 3, 5, 7};
  const model::DFlashDraftLayout draft =
      model::draftLayout(target.hiddenSize, target.vocabularySize, 1, 512, 1'000'000.0F, draftReads);
  test::writeSyntheticShard(root / "draft" / "model.safetensors", test::draftTensors(draft));
  engine::RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("flash-next-test", target, draft,
                                            {.outputHiddenSize = target.hiddenSize}, model::TargetSource::Gguf,
                                            model::VisionSource::None);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "flash-next-test";
  config.hostMemory = [] {
    return std::optional<engine::HostMemory>(engine::HostMemory{64 * engine::kGiB, 64 * engine::kGiB});
  };
  return config;
}

} // namespace splash::test
