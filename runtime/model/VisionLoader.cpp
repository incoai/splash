#include "model/VisionLoader.hpp"
#include "metal/abi/Vision.h"
#include "model/GgufFile.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace splash::model {
namespace {

using vision::Plan;
using vision::Section;

// The sections of layout, in file order, placed after the header block.
Plan plan(const ops::VisionLayout &l) {
  Plan result{l.depth, l.patchSize, 0, {}};
  auto &sections = result.sections;
  const auto add = [&](std::string mlx, std::string gguf, uint32_t rows, uint32_t columns = 1,
                       uint32_t paddedRows = 0, uint32_t paddedColumns = 0) {
    sections.push_back({"vision_tower." + mlx, std::move(gguf), rows, columns, paddedRows ? paddedRows : rows,
                        paddedColumns ? paddedColumns : columns});
  };
  const auto affine = [&](const std::string &mlx, const std::string &gguf, uint32_t rows, uint32_t columns,
                          uint32_t paddedRows = 0, uint32_t paddedColumns = 0) {
    add(mlx + ".weight", gguf + ".weight", rows, columns, paddedRows, paddedColumns);
    add(mlx + ".bias", gguf + ".bias", rows, 1, paddedRows);
  };
  const auto norm = [&](const std::string &mlx, const std::string &gguf) {
    add(mlx + ".weight", gguf + ".weight", l.hiddenSize);
    add(mlx + ".bias", gguf + ".bias", l.hiddenSize);
  };
  affine("patch_embed.proj", "v.patch_embd", l.hiddenSize, l.patchDimension);
  sections.front().patch = true;
  add("pos_embed.weight", "v.position_embd.weight", l.positionGridSide * l.positionGridSide, l.hiddenSize);
  for (uint32_t i = 0; i < l.depth; ++i) {
    const auto mlx = "blocks." + std::to_string(i) + ".", gguf = "v.blk." + std::to_string(i) + ".";
    norm(mlx + "norm1", gguf + "ln1");
    affine(mlx + "attn.qkv", gguf + "attn_qkv", 3 * l.hiddenSize, l.hiddenSize);
    affine(mlx + "attn.proj", gguf + "attn_out", l.hiddenSize, l.hiddenSize);
    norm(mlx + "norm2", gguf + "ln2");
    affine(mlx + "mlp.linear_fc1", gguf + "ffn_up", l.intermediateSize, l.hiddenSize, l.paddedIntermediateSize);
    affine(mlx + "mlp.linear_fc2", gguf + "ffn_down", l.hiddenSize, l.intermediateSize, 0, l.paddedIntermediateSize);
  }
  norm("merger.norm", "v.post_ln");
  affine("merger.linear_fc1", "mm.0", l.mergedHiddenSize, l.mergedHiddenSize);
  affine("merger.linear_fc2", "mm.2", l.outputHiddenSize, l.mergedHiddenSize);
  uint64_t at = kWeightFileAlignment;
  for (auto &s : sections) {
    s.offset = at;
    at = alignWeightOffset(at + uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  }
  result.bytes = at;
  return result;
}

// A source tensor of a dtype preparation reads.
vision::Input input(std::string name, SourceTensor tensor) {
  if (tensor.dtype != "BF16" && tensor.dtype != "F16" && tensor.dtype != "F32")
    throw WeightStoreError("vision tensor " + name + " in " + tensor.file->path().string() + " is " + tensor.dtype +
                           "; preparation reads BF16, F16 or F32");
  return {std::move(name), std::move(tensor)};
}

// A safetensors checkpoint's unquantized tower: MLX's vision_tower.*, whose
// patch embedding is one Conv3d weight [output, frame, patch-row, patch-col,
// channel], or transformers' model.visual.* (as Model Optimizer keeps it),
// whose Conv3d weight [output, channel, frame, patch-row, patch-col] is in the
// image's order already, so the writer copies it as stored.
void bindCheckpoint(const SafetensorsCheckpoint &checkpoint, const ops::VisionLayout &layout, Plan &plan) {
  const uint64_t p = layout.patchSize;
  const std::string mlxPrefix = "vision_tower.", transformersPrefix = "model.visual.";
  const bool transformers = checkpoint.find(transformersPrefix + "patch_embed.proj.weight") != nullptr;
  for (auto &s : plan.sections) {
    const std::string name = transformers ? transformersPrefix + s.mlx.substr(mlxPrefix.size()) : s.mlx;
    // A quantized module keeps its scales beside the packed weight.
    const auto stem = name.substr(0, name.rfind('.'));
    for (const auto &scales : {stem + ".scales", stem + ".weight_scale"})
      if (const SourceTensor *quantized = checkpoint.find(scales))
        throw WeightStoreError("the vision tower is quantized (" + scales + " in " + quantized->file->path().string() +
                               "); preparation needs BF16, F16 or F32 vision weights");
    const auto &tensor = checkpoint.require(name);
    const std::vector<uint64_t> shape = s.patch          ? transformers ? std::vector<uint64_t>{s.rows, 3, 2, p, p}
                                                                        : std::vector<uint64_t>{s.rows, 2, p, p, 3}
                                        : s.columns == 1 ? std::vector<uint64_t>{s.rows}
                                                         : std::vector<uint64_t>{s.rows, s.columns};
    if (tensor.shape != shape) throw WeightStoreError("vision tensor shape mismatch: " + name);
    // Transformers' patch embedding needs no reordering.
    if (transformers) s.patch = false;
    s.inputs.push_back(input(name, tensor));
  }
}

// The metadata of a clip qwen3vl_merger mmproj of this layout, with no
// deepstack block.
void requireMmprojMetadata(const GgufFile &gguf, const ops::VisionLayout &layout) {
  const GgufMetadata &metadata = gguf.metadata();
  if (metadata.architecture() != "clip" || metadata.stringValue("clip.projector_type") != "qwen3vl_merger")
    throw WeightStoreError("unsupported vision GGUF architecture");
  for (const auto &[key, expected] : std::initializer_list<std::pair<const char *, uint64_t>>{
           {"clip.vision.projection_dim", layout.outputHiddenSize},
           {"clip.vision.patch_size", layout.patchSize},
           {"clip.vision.embedding_length", layout.hiddenSize},
           {"clip.vision.feed_forward_length", layout.intermediateSize},
           {"clip.vision.block_count", layout.depth},
           {"clip.vision.attention.head_count", layout.heads},
           {"clip.vision.spatial_merge_size", layout.spatialMerge},
           {"clip.use_gelu", 1}})
    if (metadata.unsignedValue(key) != expected) throw WeightStoreError(std::string("vision metadata mismatch: ") + key);
  const auto epsilon = metadata.floatValue("clip.vision.attention.layer_norm_epsilon");
  if (!epsilon || !std::isfinite(*epsilon) || std::abs(*epsilon - SPLASH_VISION_NORM_EPSILON) > 1e-12)
    throw WeightStoreError("vision LayerNorm epsilon mismatch");
  for (const char *key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
    const auto values = gguf.numericArray(key);
    if (!values || values->size() != 3 ||
        !std::all_of(values->begin(), values->end(), [](double x) { return x == 0.5; }))
      throw WeightStoreError("vision image normalization mismatch");
  }
  // Each block must be declared: a deepstack block feeds the language model
  // through tensors this tower does not have.
  const auto deepstack = gguf.numericArray("clip.vision.is_deepstack_layers");
  if (!deepstack || deepstack->size() != layout.depth)
    throw WeightStoreError("vision metadata must list clip.vision.is_deepstack_layers per block");
  if (std::any_of(deepstack->begin(), deepstack->end(), [](double x) { return x != 0; }))
    throw WeightStoreError("vision deepstack layers are unsupported");
}

// GGUF: a qwen3vl_merger mmproj describing this tower, all of whose tensors
// preparation uses. The patch embedding is one [channel, patch-row,
// patch-col] weight per temporal frame (v.patch_embd.weight and .weight.1).
void bindMmproj(const GgufFile &gguf, const ops::VisionLayout &layout, Plan &plan) {
  requireMmprojMetadata(gguf, layout);
  const uint64_t p = layout.patchSize;
  std::set<std::string, std::less<>> used;
  for (auto &s : plan.sections) {
    for (uint32_t frame = 0; frame < (s.patch ? 2u : 1u); ++frame) {
      std::string name = s.gguf + (frame ? ".1" : "");
      const GgufTensor &t = gguf.require(name);
      const std::vector<uint64_t> shape = s.patch          ? std::vector<uint64_t>{p, p, 3, s.rows}
                                          : s.columns == 1 ? std::vector<uint64_t>{s.rows}
                                                           : std::vector<uint64_t>{s.columns, s.rows};
      if (t.dims != shape) throw WeightStoreError("vision tensor shape mismatch: " + name);
      const std::string dtype = ggmlTypeName(t.type);
      used.insert(name);
      s.inputs.push_back(input(std::move(name), {&gguf.source(), dtype, t.dims, t.offset, t.bytes}));
    }
  }
  std::string unused;
  for (const auto &t : gguf.tensors())
    if (!used.contains(t.name)) unused += (unused.empty() ? "" : ", ") + t.name;
  if (!unused.empty())
    throw WeightStoreError("mmproj tensors the vision tower does not use: " + unused + " (" +
                           gguf.source().path().string() + ")");
}

} // namespace

// The source and the plan bound to it, which the image's writer shares.
struct VisionLoader::Planned {
  ops::VisionLayout layout;
  std::unique_ptr<SafetensorsCheckpoint> checkpoint;
  std::unique_ptr<WeightSource> mmproj;
  Plan plan;
  void checkUnchanged() const {
    if (checkpoint) checkpoint->checkUnchanged();
    else mmproj->checkUnchanged();
  }
};

VisionLoader::VisionLoader(const std::filesystem::path &directory, VisionSource source,
                           const ops::VisionLayout &layout) {
  requireVisionLayout(layout);
  auto planned = std::make_shared<Planned>();
  planned->layout = layout;
  planned->plan = plan(layout);
  if (source == VisionSource::Safetensors) {
    planned->checkpoint = std::make_unique<SafetensorsCheckpoint>(directory);
    bindCheckpoint(*planned->checkpoint, layout, planned->plan);
  } else if (source == VisionSource::Gguf) {
    planned->mmproj = std::make_unique<WeightSource>(directory / "mmproj.gguf");
    bindMmproj(GgufFile(*planned->mmproj), layout, planned->plan);
  } else {
    throw WeightStoreError("only MLX and GGUF vision sources are written into images");
  }
  planned_ = std::move(planned);
}

VisionLoader::~VisionLoader() = default;

const ops::VisionLayout &VisionLoader::layout() const noexcept { return planned_->layout; }

ImagePlan VisionLoader::image() const {
  return {"vision/model.bin", std::string(kVisionMagic), planned_->layout.depth, 0, planned_->plan.bytes,
          [planned = planned_](std::span<uint8_t> bytes, const metal::MetalBuffer &) {
            vision::writeVision(bytes, planned->plan);
            planned->checkUnchanged();
          }};
}

uint64_t visionImageBytes(const ops::VisionLayout &layout) { return plan(layout).bytes; }

// The writer relies on the patch embedding's width, two frames of RGB
// patches, and on padding that only adds rows or columns.
void requireVisionLayout(const ops::VisionLayout &layout) {
  if (!layout.depth || !layout.hiddenSize || !layout.patchDimension || !layout.intermediateSize ||
      !layout.paddedIntermediateSize || !layout.mergedHiddenSize || !layout.outputHiddenSize || !layout.heads ||
      !layout.headDimension || !layout.positionGridSide || !layout.patchSize || !layout.spatialMerge ||
      layout.heads * layout.headDimension != layout.hiddenSize ||
      layout.paddedIntermediateSize < layout.intermediateSize ||
      layout.mergedHiddenSize != layout.hiddenSize * layout.spatialMerge * layout.spatialMerge ||
      layout.patchDimension != 3 * 2 * layout.patchSize * layout.patchSize)
    throw WeightStoreError("Qwen vision layout is inconsistent");
}

} // namespace splash::model
