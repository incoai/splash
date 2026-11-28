#include "model/InklingVision.hpp"

#include "Checked.hpp"
#include "metal/abi/Vision.h"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

using vision::Plan;
using vision::Section;

// A section of the image and the mmproj tensors it is written from: none for
// a zero bias, several stacked for q, k and v and the class token.
struct Planned {
  Section section;
  std::vector<std::string> tensors;
};

// The image's sections in file order, each 16 KiB aligned after the header
// block.
std::vector<Planned> sections(const ops::VisionLayout &l) {
  std::vector<Planned> result;
  const auto add = [&](std::vector<std::string> tensors, std::string label, uint32_t rows, uint32_t columns = 1,
                       uint32_t storedColumns = 0, bool projection = false) {
    const bool stacked = tensors.size() > 1;
    result.push_back({{.mlx = {},
                       .gguf = std::move(label),
                       .rows = rows,
                       .columns = columns,
                       .storedRows = rows,
                       .storedColumns = storedColumns ? storedColumns : columns,
                       .stacked = stacked,
                       .projection = projection},
                      std::move(tensors)});
  };
  // A block's or the merger's projection, which may be stored as Q8_0
  // planes (vision::placeQ8Matrices).
  const auto projection = [&](std::vector<std::string> tensors, std::string label, uint32_t rows, uint32_t columns) {
    add(std::move(tensors), std::move(label), rows, columns, 0, true);
  };
  const auto norm = [&](const std::string &name) {
    add({name + ".weight"}, name + ".weight", l.hiddenSize);
    add({name + ".bias"}, name + ".bias", l.hiddenSize);
  };
  add({"v.patch_embd.weight"}, "v.patch_embd.weight", l.hiddenSize, l.patchDimension, l.paddedPatchDimension);
  add({}, "patch embedding bias", l.hiddenSize);
  add({"v.position_embd.weight"}, "v.position_embd.weight", l.positionGridSide * l.positionGridSide, l.hiddenSize);
  add({"v.class_embd", "v.class_position_embd.weight"}, "class token", 2, l.hiddenSize);
  norm("v.pre_ln");
  for (uint32_t i = 0; i < l.depth; ++i) {
    const std::string block = "v.blk." + std::to_string(i) + ".";
    norm(block + "ln1");
    projection({block + "attn_q.weight", block + "attn_k.weight", block + "attn_v.weight"},
               block + "attn_qkv.weight", 3 * l.hiddenSize, l.hiddenSize);
    add({block + "attn_q.bias", block + "attn_k.bias", block + "attn_v.bias"}, block + "attn_qkv.bias",
        3 * l.hiddenSize);
    projection({block + "attn_out.weight"}, block + "attn_out.weight", l.hiddenSize, l.hiddenSize);
    add({block + "attn_out.bias"}, block + "attn_out.bias", l.hiddenSize);
    add({block + "ls1.weight"}, block + "ls1.weight", l.hiddenSize);
    norm(block + "ln2");
    projection({block + "ffn_up.weight"}, block + "ffn_up.weight", l.intermediateSize, l.hiddenSize);
    add({block + "ffn_up.bias"}, block + "ffn_up.bias", l.intermediateSize);
    projection({block + "ffn_down.weight"}, block + "ffn_down.weight", l.hiddenSize, l.intermediateSize);
    add({block + "ffn_down.bias"}, block + "ffn_down.bias", l.hiddenSize);
    add({block + "ls2.weight"}, block + "ls2.weight", l.hiddenSize);
  }
  projection({"mm.1.weight"}, "mm.1.weight", l.mergedHiddenSize, l.mergedHiddenSize);
  add({}, "mm.1.bias", l.mergedHiddenSize);
  projection({"mm.2.weight"}, "mm.2.weight", l.outputHiddenSize, l.mergedHiddenSize);
  add({}, "mm.2.bias", l.outputHiddenSize);
  return result;
}

// The metadata of a clip inkling2v mmproj of this layout: its sizes,
// LayerNorm epsilon and normalization (mean and deviation 0.5, x / 127.5 -
// 1 on the 0-255 scale), and the image size its position grid covers.
void requireMetadata(const GgufFile &gguf, const ops::VisionLayout &layout) {
  const GgufMetadata &metadata = gguf.metadata();
  if (metadata.architecture() != "clip" || metadata.stringValue("clip.vision.projector_type") != "inkling2v")
    throw WeightStoreError("unsupported vision GGUF architecture");
  for (const auto &[key, expected] : std::initializer_list<std::pair<const char *, uint64_t>>{
           {"clip.vision.projection_dim", layout.outputHiddenSize},
           {"clip.vision.patch_size", layout.patchSize},
           {"clip.vision.embedding_length", layout.hiddenSize},
           {"clip.vision.feed_forward_length", layout.intermediateSize},
           {"clip.vision.block_count", layout.depth},
           {"clip.vision.attention.head_count", layout.heads},
           {"clip.vision.spatial_merge_size", layout.spatialMerge},
           {"clip.vision.image_size", uint64_t{layout.positionGridSide} * layout.patchSize}})
    if (metadata.unsignedValue(key) != expected) throw WeightStoreError(std::string("vision metadata mismatch: ") + key);
  // GGUF states the epsilon as an F32.
  const auto epsilon = metadata.floatValue("clip.vision.attention.layer_norm_epsilon");
  if (!epsilon || !std::isfinite(*epsilon) ||
      static_cast<float>(*epsilon) != static_cast<float>(SPLASH_INKLING_VISION_NORM_EPSILON))
    throw WeightStoreError("vision LayerNorm epsilon mismatch");
  for (const char *key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
    const auto values = gguf.numericArray(key);
    if (!values || values->size() != 3 ||
        !std::all_of(values->begin(), values->end(), [](double x) { return x == 0.5; }))
      throw WeightStoreError("vision image normalization mismatch");
  }
}

// The shape a tensor of the section has in GGUF order (row length first):
// the patch embedding's [patch-col, patch-row, channel, output], a stacked
// part's rows of the section's, a vector's.
std::vector<uint64_t> tensorShape(const Section &s, uint32_t parts, uint32_t patchSize) {
  if (s.gguf == "v.patch_embd.weight") return {patchSize, patchSize, 3, s.rows};
  const uint64_t rows = s.rows / parts;
  if (s.columns == 1) return {rows};
  if (rows == 1) return {s.columns};
  return {s.columns, rows};
}

// How a safetensors checkpoint names the tower's tensors: transformers'
// Inkling2VisionModel's under model.vision_tower. (an MLX conversion's, a
// fine-tune transformers saved), the release's own, its backbone under
// model.visual.backbone. beside its projector's two matrices
// model.visual.proj_in and proj_out, or either with the release's
// model.visual. renamed vision_tower. (oMLX's keep the backbone's names; an
// nvfp4 conversion gives it transformers' instead, without its backbone. and
// merger).
struct CheckpointNames {
  std::string_view tower;     // the backbone's prefix
  std::string_view projector; // proj_in's and proj_out's, or empty for the merger's
  bool release;               // the release's block names (ln_1, ls_1.gamma, mlp.c_fc)
  bool positionWeight;        // position_embedding.weight, not position_embedding
  std::string_view owner;     // the prefix of every tensor of the tower's
};
constexpr CheckpointNames kCheckpointNames[] = {
    {"model.vision_tower.", "", false, true, "model.vision_tower."},
    {"model.visual.backbone.", "model.visual.", true, false, "model.visual."},
    {"vision_tower.backbone.", "vision_tower.", true, false, "vision_tower."},
    {"vision_tower.", "vision_tower.", false, false, "vision_tower."},
};

// The names of the checkpoint's tower: those whose patch embedding it holds,
// else transformers', which a refusal then names.
const CheckpointNames &checkpointNames(const SafetensorsCheckpoint &checkpoint) {
  for (const CheckpointNames &names : kCheckpointNames)
    if (checkpoint.find(std::string(names.tower) + "embeddings.patch_embedding.weight")) return names;
  return kCheckpointNames[0];
}

// The name the checkpoint gives the tower's tensor its mmproj names `gguf`.
std::string checkpointName(const std::string &gguf, const CheckpointNames &names) {
  const bool release = names.release;
  const std::string tower(names.tower), projector(names.projector);
  const std::string embeddings = tower + "embeddings.";
  if (gguf == "v.patch_embd.weight") return embeddings + "patch_embedding.weight";
  if (gguf == "v.position_embd.weight")
    return embeddings + (names.positionWeight ? "position_embedding.weight" : "position_embedding");
  if (gguf == "v.class_embd") return embeddings + "class_embedding";
  if (gguf == "v.class_position_embd.weight") return embeddings + "cls_position_embedding";
  if (gguf == "mm.1.weight") return projector.empty() ? tower + "merger.linear_1.weight" : projector + "proj_in.weight";
  if (gguf == "mm.2.weight") return projector.empty() ? tower + "merger.linear_2.weight" : projector + "proj_out.weight";
  const auto suffix = gguf.substr(gguf.rfind('.'));  // .weight or .bias
  const auto stem = gguf.substr(0, gguf.size() - suffix.size());
  if (stem == "v.pre_ln") return tower + (release ? "ln_pre" : "pre_layernorm") + suffix;
  // v.blk.N.PART
  const std::string blockPrefix = "v.blk.";
  if (stem.rfind(blockPrefix, 0) != 0) throw std::logic_error("unknown Inkling vision tensor " + gguf);
  const auto dot = stem.find('.', blockPrefix.size());
  const std::string layer = tower + "layers." + stem.substr(blockPrefix.size(), dot - blockPrefix.size()) + ".";
  const std::string part = stem.substr(dot + 1);
  if (part == "ln1") return layer + (release ? "ln_1" : "layernorm_before") + suffix;
  if (part == "ln2") return layer + (release ? "ln_2" : "layernorm_after") + suffix;
  if (part == "attn_q" || part == "attn_k" || part == "attn_v")
    return layer + "self_attn." + part.substr(5) + "_proj" + suffix;
  if (part == "attn_out") return layer + "self_attn.out_proj" + suffix;
  if (part == "ffn_up") return layer + (release ? "mlp.c_fc" : "mlp.fc1") + suffix;
  if (part == "ffn_down") return layer + (release ? "mlp.c_proj" : "mlp.fc2") + suffix;
  // A LayerScale is a vector of its own: ls1.weight and ls2.weight in an mmproj.
  if (part == "ls1") return layer + (release ? "ls_1.gamma" : "lambda_1");
  if (part == "ls2") return layer + (release ? "ls_2.gamma" : "lambda_2");
  throw std::logic_error("unknown Inkling vision tensor " + gguf);
}

// The row-major shape a checkpoint gives a tensor of the section: a stacked
// part's rows of the section's, a vector's.
std::vector<uint64_t> checkpointShape(const Section &s, uint32_t parts) {
  const uint64_t rows = s.rows / parts;
  if (s.columns == 1) return {rows};
  if (rows == 1) return {s.columns};
  return {rows, s.columns};
}

} // namespace

Plan inklingVisionPlan(const ops::VisionLayout &layout) {
  Plan result{layout.depth, layout.patchSize, 0, {}};
  for (auto &[section, tensors] : sections(layout)) result.sections.push_back(std::move(section));
  vision::placeSections(result);
  return result;
}

void bindInklingMmproj(const GgufFile &gguf, const ops::VisionLayout &layout, Plan &plan) {
  requireMetadata(gguf, layout);
  const std::vector<Planned> planned = sections(layout);
  if (planned.size() != plan.sections.size()) throw std::logic_error("Inkling vision plan mismatch");
  std::set<std::string, std::less<>> used;
  for (size_t index = 0; index < planned.size(); ++index) {
    Section &s = plan.sections[index];
    const std::vector<std::string> &tensors = planned[index].tensors;
    const auto shape = tensorShape(s, static_cast<uint32_t>(std::max<size_t>(tensors.size(), 1)), layout.patchSize);
    for (const std::string &name : tensors) {
      const GgufTensor &t = gguf.require(name);
      if (t.dims != shape) throw WeightStoreError("vision tensor shape mismatch: " + name);
      const std::string dtype = ggmlTypeName(t.type);
      const auto refusal = [&](const std::string &reads) {
        return WeightStoreError("vision tensor " + name + " in " + gguf.source().path().string() + " is " + dtype +
                                "; preparation reads " + reads);
      };
      if (dtype != "BF16" && dtype != "F16" && dtype != "F32" && dtype != "Q8_0")
        throw refusal("BF16, F16, F32 or Q8_0");
      // A Q8_0 block holds 32 values of a row and the writer converts whole
      // blocks, so a row is whole blocks: not a vector's, of one value each.
      if (dtype == "Q8_0" && s.columns % 32)
        throw refusal("Q8_0 only in a matrix's rows of whole blocks of 32 values");
      used.insert(name);
      s.inputs.push_back({name, {&gguf.source(), dtype, t.dims, t.offset, t.bytes}});
    }
  }
  std::string unused;
  for (const auto &t : gguf.tensors())
    if (!used.contains(t.name)) unused += (unused.empty() ? "" : ", ") + t.name;
  if (!unused.empty())
    throw WeightStoreError("mmproj tensors the vision tower does not use: " + unused + " (" +
                           gguf.source().path().string() + ")");
  vision::placeQ8Matrices(plan);
}

void bindInklingCheckpoint(const SafetensorsCheckpoint &checkpoint, const ops::VisionLayout &layout, Plan &plan) {
  const CheckpointNames &names = checkpointNames(checkpoint);
  const std::vector<Planned> planned = sections(layout);
  if (planned.size() != plan.sections.size()) throw std::logic_error("Inkling vision plan mismatch");
  std::set<std::string, std::less<>> used;
  for (size_t index = 0; index < planned.size(); ++index) {
    Section &s = plan.sections[index];
    const std::vector<std::string> &tensors = planned[index].tensors;
    const auto shape = checkpointShape(s, static_cast<uint32_t>(std::max<size_t>(tensors.size(), 1)));
    for (const std::string &gguf : tensors) {
      const std::string name = checkpointName(gguf, names);
      // A quantized module keeps its scales beside the packed weight.
      const auto stem = name.substr(0, name.rfind('.'));
      for (const auto &scales : {stem + ".scales", stem + ".weight_scale"})
        if (const SourceTensor *quantized = checkpoint.find(scales))
          throw WeightStoreError("the vision tower is quantized (" + scales + " in " +
                                 quantized->file->path().string() + "); preparation needs BF16, F16 or F32 vision weights");
      const SourceTensor &tensor = checkpoint.require(name);
      if (tensor.shape != shape) throw WeightStoreError("vision tensor shape mismatch: " + name);
      if (tensor.dtype != "BF16" && tensor.dtype != "F16" && tensor.dtype != "F32")
        throw WeightStoreError("vision tensor " + name + " in " + tensor.file->path().string() + " is " + tensor.dtype +
                               "; preparation reads BF16, F16 or F32");
      used.insert(name);
      s.inputs.push_back({name, tensor});
    }
    // Its linear patch embedding orders a patch's values [patch-row,
    // patch-col, channel], which the writer reorders to the image's.
    if (s.gguf == "v.patch_embd.weight") s.patch = true;
  }
  // Every tensor of the tower's is one it binds, as an mmproj's is: a bias
  // or a layer the tower does not take would change what it computes.
  std::string unused;
  for (const std::string &name : checkpoint.names(names.owner))
    if (!used.contains(name)) unused += (unused.empty() ? "" : ", ") + name;
  if (!unused.empty()) throw WeightStoreError("checkpoint tensors the vision tower does not use: " + unused);
}

ops::VisionWeights readInklingVision(WeightFile &file, const ops::VisionLayout &layout, const Plan &plan) {
  const auto bytes = [](uint64_t rows, uint64_t columns) {
    return checkedMultiply<WeightStoreError>(checkedMultiply<WeightStoreError>(rows, columns, "vision weight elements"),
                                             kBFloat16Bytes, "vision weight bytes");
  };
  // The plan's projections stored as Q8_0 planes, by their section's name.
  std::set<std::string, std::less<>> q8;
  for (const Section &s : plan.sections)
    if (s.q8Planes) q8.insert(s.gguf);
  const auto affine = [&](uint32_t output, uint32_t input, std::string_view label,
                          std::string_view section = {}) -> ops::DenseWeights {
    ops::DenseWeights weights;
    if (q8.contains(section)) {
      weights.quantized = readQuantizedSegment(file, label);
      if (weights.quantized.formatId != GGUF_FMT_Q80 || weights.quantized.outputSize != output ||
          weights.quantized.inputSize != input)
        throw WeightStoreError("vision Q8_0 matrix does not match the layout: " + std::string(label));
    } else {
      weights.weight = file.section(bytes(output, input), label);
    }
    weights.bias = file.section(bytes(output, 1), label);
    return weights;
  };
  const auto norm = [&](std::string_view label) -> ops::LayerNormWeights {
    return {file.section(bytes(layout.hiddenSize, 1), label), file.section(bytes(layout.hiddenSize, 1), label)};
  };
  ops::VisionWeights result;
  result.layout = layout;
  result.patchEmbedding = affine(layout.hiddenSize, layout.paddedPatchDimension, "patch-embed");
  result.positionTable =
      file.section(bytes(uint64_t{layout.positionGridSide} * layout.positionGridSide, layout.hiddenSize),
                   "position-table");
  result.classToken = file.section(bytes(2, layout.hiddenSize), "class-token");
  result.inputNorm = norm("input-norm");
  result.blocks.reserve(layout.depth);
  for (uint32_t index = 0; index < layout.depth; ++index) {
    const std::string name = "v.blk." + std::to_string(index) + ".";
    ops::VisionBlock block;
    block.norm1 = norm("norm1");
    block.qkv = affine(3 * layout.hiddenSize, layout.hiddenSize, "qkv", name + "attn_qkv.weight");
    // Each output projection's LayerScale follows its bias.
    block.projection = affine(layout.hiddenSize, layout.hiddenSize, "proj", name + "attn_out.weight");
    block.projection.scale = file.section(bytes(layout.hiddenSize, 1), "ls1");
    block.norm2 = norm("norm2");
    block.upProjection = affine(layout.intermediateSize, layout.hiddenSize, "fc1", name + "ffn_up.weight");
    block.downProjection = affine(layout.hiddenSize, layout.intermediateSize, "fc2", name + "ffn_down.weight");
    block.downProjection.scale = file.section(bytes(layout.hiddenSize, 1), "ls2");
    result.blocks.push_back(std::move(block));
  }
  result.mergerUpProjection = affine(layout.mergedHiddenSize, layout.mergedHiddenSize, "merger-fc1", "mm.1.weight");
  result.mergerDownProjection =
      affine(layout.outputHiddenSize, layout.mergedHiddenSize, "merger-fc2", "mm.2.weight");
  return result;
}

} // namespace splash::model
