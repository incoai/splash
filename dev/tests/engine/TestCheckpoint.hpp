#pragma once

// Synthetic installed models: the target/, draft/ and vision/ directories of
// an MLX model of given layouts, each a safetensors shard holding every tensor
// its loader reads, all zero, which loadModel loads as it loads an installed
// model.

#include "TestChecks.hpp"
#include "metal/abi/DraftAttention.h"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/MlxTarget.hpp"
#include "model/VisionLoader.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace splash::test {

struct SyntheticTensor final {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> shape;
};

// A safetensors shard of tensors whose values are all zero: the data is a
// hole in the file, so a large model costs no disk.
inline void writeSyntheticShard(const std::filesystem::path &path,
                                const std::vector<SyntheticTensor> &tensors) {
  std::string header = "{";
  uint64_t offset = 0;
  for (const SyntheticTensor &tensor : tensors) {
    uint64_t bytes = tensor.dtype == "U32" || tensor.dtype == "F32" ? 4 : tensor.dtype == "U8" ? 1 : 2;
    std::string shape;
    for (uint64_t dimension : tensor.shape) {
      bytes *= dimension;
      shape += (shape.empty() ? "" : ",") + std::to_string(dimension);
    }
    header += (header.size() > 1 ? ",\"" : "\"") + tensor.name + R"(":{"dtype":")" + tensor.dtype +
              R"(","shape":[)" + shape + R"(],"data_offsets":[)" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  std::filesystem::create_directories(path.parent_path());
  {
    std::ofstream file(path, std::ios::binary);
    const uint64_t length = header.size();
    file.write(reinterpret_cast<const char *>(&length), sizeof length);
    file << header;
    require(bool(file), "unable to write a synthetic safetensors shard");
  }
  std::filesystem::resize_file(path, sizeof(uint64_t) + header.size() + offset);
}

// The tensors of a DFlash2 draft of layout as its repository releases them,
// every one BF16, which model/DraftCheckpoint.cpp reads.
inline std::vector<SyntheticTensor> draftTensors(const model::DFlashDraftLayout &layout) {
  std::vector<SyntheticTensor> result;
  const auto add = [&](const std::string &name, std::vector<uint64_t> shape) {
    result.push_back({name, "BF16", std::move(shape)});
  };
  const uint64_t hidden = layout.hiddenSize, kv = uint64_t{layout.kvHeads} * layout.attentionHeadDimension;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    const std::string attention = prefix + "self_attn.";
    add(prefix + "input_layernorm.weight", {hidden});
    for (const std::string convolution : {"attention_conv.", "mlp_conv."}) {
      add(prefix + convolution + "base_kernel",
          {SPLASH_DRAFT_CONVOLUTION_STAGES, SPLASH_DRAFT_CONVOLUTION_TAPS, hidden});
      add(prefix + convolution + "kernel_projection.weight", {layout.dynamicSize, hidden});
    }
    add(attention + "q_proj.weight", {layout.attentionSize, hidden});
    add(attention + "k_proj.weight", {kv, hidden});
    add(attention + "v_proj.weight", {kv, hidden});
    add(attention + "q_norm.weight", {layout.attentionHeadDimension});
    add(attention + "k_norm.weight", {layout.attentionHeadDimension});
    add(attention + "o_proj.weight", {hidden, layout.attentionSize});
    add(prefix + "post_attention_layernorm.weight", {hidden});
    add(prefix + "mlp.gate_proj.weight", {layout.intermediateSize, hidden});
    add(prefix + "mlp.up_proj.weight", {layout.intermediateSize, hidden});
    add(prefix + "mlp.down_proj.weight", {hidden, layout.intermediateSize});
  }
  add("fc.weight", {hidden, layout.targetHiddenSize});
  add("hidden_norm.weight", {hidden});
  add("norm.weight", {hidden});
  add("candidate_selector.hidden_projection.weight", {layout.selectorRank, hidden});
  add("candidate_selector.predecessor_codebook", {layout.vocabularySize, layout.selectorRank});
  add("candidate_selector.successor_codebook", {layout.vocabularySize, layout.selectorRank});
  return result;
}

// The tensors of an MLX target of layout as mlx-lm saves it, which
// model/MlxImage.cpp reads: bf16 norms, convolution, A_log and dt_bias, and
// each quantized module's codes, scales and biases in the bits and group size
// quantization(module) gives, as MLX packs them.
template <class Layout, class Quantization>
std::vector<SyntheticTensor> mlxTargetTensors(const Layout &layout, Quantization quantization) {
  std::vector<SyntheticTensor> result;
  const auto bfloat16 = [&](const std::string &name, std::vector<uint64_t> shape) {
    result.push_back({name, "BF16", std::move(shape)});
  };
  const auto quantized = [&](const std::string &module, uint64_t rows, uint64_t columns, uint64_t experts = 1) {
    const auto [bits, group] = quantization(module);
    const auto shape = [&](uint64_t last) {
      return experts > 1 ? std::vector<uint64_t>{experts, rows, last} : std::vector<uint64_t>{rows, last};
    };
    result.push_back({module + ".weight", "U32", shape(columns * bits / 32)});
    bfloat16(module + ".scales", shape(columns / group));
    bfloat16(module + ".biases", shape(columns / group));
  };
  const uint64_t hidden = layout.hiddenSize;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
    bfloat16(prefix + "input_layernorm.weight", {hidden});
    if (layout.isFullAttentionLayer(layer)) {
      const std::string attention = prefix + "self_attn.";
      const uint64_t kv = uint64_t{layout.attentionKvHeads} * layout.attentionHeadDimension;
      quantized(attention + "q_proj", 2ull * layout.attentionWidth, hidden);
      quantized(attention + "k_proj", kv, hidden);
      quantized(attention + "v_proj", kv, hidden);
      bfloat16(attention + "q_norm.weight", {layout.attentionHeadDimension});
      bfloat16(attention + "k_norm.weight", {layout.attentionHeadDimension});
      quantized(attention + "o_proj", hidden, layout.attentionWidth);
    } else {
      const std::string gdn = prefix + "linear_attn.";
      quantized(gdn + "in_proj_qkv", layout.convolutionDimension, hidden);
      quantized(gdn + "in_proj_z", layout.attentionWidth, hidden);
      quantized(gdn + "in_proj_b", layout.gdnValueHeads, hidden);
      quantized(gdn + "in_proj_a", layout.gdnValueHeads, hidden);
      bfloat16(gdn + "conv1d.weight", {layout.convolutionDimension, model::kGdnConvolutionTaps, 1});
      bfloat16(gdn + "A_log", {layout.gdnValueHeads});
      bfloat16(gdn + "dt_bias", {layout.gdnValueHeads});
      bfloat16(gdn + "norm.weight", {layout.gdnHeadDimension});
      quantized(gdn + "out_proj", hidden, layout.attentionWidth);
    }
    bfloat16(prefix + "post_attention_layernorm.weight", {hidden});
    const std::string mlp = prefix + "mlp.";
    const auto ffn = [&](const std::string &projections, uint64_t width, uint64_t experts) {
      quantized(projections + "gate_proj", width, hidden, experts);
      quantized(projections + "up_proj", width, hidden, experts);
      quantized(projections + "down_proj", hidden, width, experts);
    };
    if (layout.ffnKind == model::QwenFfnKind::SparseMoe) {
      quantized(mlp + "gate", layout.experts, hidden);
      ffn(mlp + "switch_mlp.", layout.expertIntermediateSize, layout.experts);
      ffn(mlp + "shared_expert.", layout.expertIntermediateSize, 1);
      quantized(mlp + "shared_expert_gate", 1, hidden);
    } else {
      ffn(mlp, layout.intermediateSize, 1);
    }
  }
  bfloat16("language_model.model.norm.weight", {hidden});
  quantized("language_model.lm_head", layout.vocabularySize, hidden);
  quantized("language_model.model.embed_tokens", layout.vocabularySize, hidden);
  return result;
}

// The MLX vision tower's tensors of layout, as model/VisionLoader.cpp reads
// them: the patch embedding over two frames of RGB patches, the position
// table, then each block's and the merger's norms and projections.
inline std::vector<SyntheticTensor> visionTensors(const ops::VisionLayout &layout) {
  std::vector<SyntheticTensor> result;
  const auto add = [&](const std::string &name, std::vector<uint64_t> shape) {
    result.push_back({"vision_tower." + name, "BF16", std::move(shape)});
  };
  const auto affine = [&](const std::string &name, uint64_t rows, uint64_t columns) {
    add(name + ".weight", {rows, columns});
    add(name + ".bias", {rows});
  };
  const auto norm = [&](const std::string &name) {
    add(name + ".weight", {layout.hiddenSize});
    add(name + ".bias", {layout.hiddenSize});
  };
  add("patch_embed.proj.weight", {layout.hiddenSize, 2, layout.patchSize, layout.patchSize, 3});
  add("patch_embed.proj.bias", {layout.hiddenSize});
  add("pos_embed.weight", {uint64_t{layout.positionGridSide} * layout.positionGridSide, layout.hiddenSize});
  for (uint32_t block = 0; block < layout.depth; ++block) {
    const std::string at = "blocks." + std::to_string(block) + ".";
    norm(at + "norm1");
    affine(at + "attn.qkv", 3ull * layout.hiddenSize, layout.hiddenSize);
    affine(at + "attn.proj", layout.hiddenSize, layout.hiddenSize);
    norm(at + "norm2");
    affine(at + "mlp.linear_fc1", layout.intermediateSize, layout.hiddenSize);
    affine(at + "mlp.linear_fc2", layout.hiddenSize, layout.intermediateSize);
  }
  norm("merger.norm");
  affine("merger.linear_fc1", layout.mergedHiddenSize, layout.mergedHiddenSize);
  affine("merger.linear_fc2", layout.outputHiddenSize, layout.mergedHiddenSize);
  return result;
}

// The bytes of each role's images.
struct SyntheticAccounting final {
  uint64_t targetBytes = 0;
  uint64_t draftBytes = 0;
  uint64_t visionBytes = 0;
};

// Writes the target/ and draft/ checkpoints of an installed MLX model of these
// layouts under root, which weight planning reads before any image loads
// (modelWeightBytes): the target 4-bit in groups of 64 with a MoE's router and
// shared-expert gate 8-bit, as mlx-community's 4-bit releases keep them, and
// the DFlash2 draft.
template <class Layout>
void writeSyntheticCheckpoints(const std::filesystem::path &root, const Layout &target,
                               const model::DFlashDraftLayout &draft) {
  const auto releaseBits = [](const std::string &module) {
    return module.ends_with(".gate") || module.ends_with("shared_expert_gate") ? std::pair{8u, 64u}
                                                                                : std::pair{4u, 64u};
  };
  writeSyntheticShard(root / "target" / "model.safetensors", mlxTargetTensors(target, releaseBits));
  writeSyntheticShard(root / "draft" / "model.safetensors", draftTensors(draft));
}

// Writes an installed MLX model of these layouts under root: its checkpoints
// (writeSyntheticCheckpoints) and vision tower. Returns the bytes of the
// images it loads into.
template <class Layout>
SyntheticAccounting writeSyntheticModel(const std::filesystem::path &root, const Layout &target,
                                        const model::DFlashDraftLayout &draft,
                                        const ops::VisionLayout &vision) {
  writeSyntheticCheckpoints(root, target, draft);
  writeSyntheticShard(root / "vision" / "model.safetensors", visionTensors(vision));
  return {model::mlxTargetImageBytes(root / "target", target), model::draftImageBytes(root / "draft", draft),
          model::visionImageBytes(vision)};
}

} // namespace splash::test
