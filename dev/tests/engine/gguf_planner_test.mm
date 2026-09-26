// The GGUF image planner on the small dense and MoE targets and a rotated
// dense one: where each tensor goes and in which row order, which F32 tensors
// it narrows to bf16, and the sources it refuses.
//   gguf-planner
#include "GgufFixtures.hpp"
#include "metal/abi/Gguf.h"

#include <optional>

using namespace gguf_fixtures;

namespace {

// The images of the GGUF at path, planned for geometry; the planner's error
// when it refuses the file.
struct Plan {
  std::vector<model::gguf::Image> images;
  std::optional<std::string> error;
};

Plan plan(const std::filesystem::path &path, const model::gguf::TargetGeometry &geometry) {
  try {
    model::WeightSource source(path);
    const model::GgufFile gguf(source);
    return {model::gguf::planImages(gguf, geometry), std::nullopt};
  } catch (const model::GgufError &error) {
    return {{}, error.what()};
  }
}

// Whether a refusal names every part.
bool names(const Plan &result, std::initializer_list<std::string_view> parts) {
  return result.error && std::all_of(parts.begin(), parts.end(), [&](std::string_view part) {
           return result.error->find(part) != std::string::npos;
         });
}

// The plan's copy of the named tensor in any image, or nullptr.
const model::gguf::Copy *copyIn(const Plan &result, const std::string &name) {
  for (const model::gguf::Image &image : result.images)
    if (const auto *copy = copyOf(image, name)) return copy;
  return nullptr;
}

// Value heads regrouped from llama.cpp's tiled order from row `from` on.
bool grouped(const model::gguf::RowOrder &order, uint64_t from, uint32_t headRows,
             const model::gguf::TargetGeometry &g) {
  return order.from == from && order.headRows == headRows && order.keyHeads == g.gdnKeyHeads &&
         order.valueHeadsPerKey == g.gdnValueHeads / g.gdnKeyHeads;
}

// A copy of the tensor's rows as the GGUF stores them.
bool asStored(const model::gguf::Copy *copy, const test_gguf::Bytes &data) {
  return copy && !copy->bfloat16 && copy->source.order.from == UINT64_MAX &&
         copy->source.rows * copy->source.rowBytes == data.size();
}

void checkDense(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(false);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "dense.gguf";
  writeGguf(path, target.tensors, g);
  const Plan dense = plan(path, g);
  check(!dense.error, "planner plans the dense target" + (dense.error ? ": " + *dense.error : ""));
  if (dense.error) return;
  const model::gguf::Image &gdn = dense.images[0];

  // One 256-row Q8_0 tensor of the beta then the alpha rows, each in grouped
  // head order.
  const auto *alphaBeta = repackOf(gdn, "blk.0.ssm_beta.weight");
  const auto heads = [&](const model::gguf::TensorRows &rows, const char *name) {
    return rows.name == name && rows.rows == g.gdnValueHeads && grouped(rows.order, 0, 1, g);
  };
  check(alphaBeta && alphaBeta->format == GGUF_FMT_Q80 && alphaBeta->rows == QUANT_TILE_ROWS &&
            alphaBeta->sources.size() == 2 && heads(alphaBeta->sources[0], "blk.0.ssm_beta.weight") &&
            heads(alphaBeta->sources[1], "blk.0.ssm_alpha.weight"),
        "planner alpha/beta: one 256-row Q8_0 tensor of beta then alpha rows in grouped order");

  for (const Tensor &tensor : target.tensors)
    if (isNorm(tensor.name))
      check(asStored(copyIn(dense, tensor.name), tensor.data), "planner keeps F32 values as stored: " + tensor.name);

  const uint32_t keyRows = g.convolutionDimension - g.gdnValueHeads * g.gdnHeadDimension;
  const auto *convolution = copyOf(gdn, "blk.0.ssm_conv1d.weight");
  check(convolution && convolution->bfloat16 && convolution->source.rows == g.convolutionDimension &&
            grouped(convolution->source.order, keyRows, g.gdnHeadDimension, g),
        "planner narrows the convolution to bf16 and groups its value heads");
  const auto *decay = copyOf(gdn, "blk.0.ssm_a");
  check(decay && !decay->bfloat16 && grouped(decay->source.order, 0, 1, g),
        "planner keeps the decay F32 and groups its value heads");
  const auto *bias = copyOf(gdn, "blk.0.ssm_dt.bias");
  check(bias && bias->bfloat16 && grouped(bias->source.order, 0, 1, g),
        "planner narrows the time bias to bf16 and groups its value heads");

  // A key or value projection of other than attentionKvHeads heads.
  for (const std::string name : {"blk.1.attn_k.weight", "blk.1.attn_v.weight"}) {
    for (uint64_t rows : {uint64_t{g.attentionHeadDimension}, uint64_t{3} * g.attentionHeadDimension}) {
      std::vector<Tensor> malformed = target.tensors;
      Tensor &t = tensorNamed(malformed, name);
      t.dims[1] = rows;
      t.data.resize(rows * gguf_reference::rowBytes(Fmt(gguf_format_of(t.type)), g.hiddenSize));
      writeGguf(path, malformed, g);
      check(names(plan(path, g), {"shape", name}),
            "planner refuses mismatched KV rows by name: " + name + " rows=" + std::to_string(rows));
    }
  }
}

void checkMoe(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(true);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "moe.gguf";

  model::gguf::TargetGeometry wrongExperts = g;
  wrongExperts.experts = 5;
  writeGguf(path, target.tensors, wrongExperts);
  check(names(plan(path, g), {"expert_count"}), "planner names a metadata mismatch");
  model::gguf::TargetGeometry dense = g;
  dense.experts = 0;
  dense.intermediateSize = g.expertIntermediateSize;
  writeGguf(path, target.tensors, dense);
  check(names(plan(path, g), {"architecture", dense.architecture(), g.architecture()}),
        "planner checks the architecture against the package");

  writeGguf(path, target.tensors, g);
  const Plan moe = plan(path, g);
  check(!moe.error, "planner plans the MoE target" + (moe.error ? ": " + *moe.error : ""));
  if (moe.error) return;
  const model::gguf::Image &layer = moe.images[0];
  const auto *beta = copyOf(layer, "blk.0.ssm_beta.weight"), *alpha = copyOf(layer, "blk.0.ssm_alpha.weight");
  check(beta && alpha && alpha->destination == beta->destination + target.data(beta->source.name).size() &&
            grouped(beta->source.order, 0, 1, g) && grouped(alpha->source.order, 0, 1, g) && !beta->bfloat16 &&
            !alpha->bfloat16,
        "planner F32 alpha/beta tensor: beta then alpha rows in grouped order");
  for (const std::string name : {"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp_shexp.weight"})
    check(asStored(copyOf(layer, name), target.data(name)), "planner copies the F32 tensor as stored: " + name);

  // Each quantized tensor is repacked on its own, a 3-D expert tensor as one
  // tensor of experts * N rows.
  size_t quantized = 0;
  for (const Tensor &tensor : target.tensors) {
    if (!tensor.name.starts_with("blk.0.") || gguf_format_of(tensor.type) == GGUF_FMT_COUNT) continue;
    ++quantized;
    const uint64_t rows = tensor.dims[1] * (tensor.dims.size() > 2 ? tensor.dims[2] : 1);
    const auto *repack = repackOf(layer, tensor.name);
    check(repack && repack->sources.size() == 1 && repack->rows == rows && repack->columns == tensor.dims[0] &&
              repack->sources[0].rows == rows,
          "planner repacks " + tensor.name + " as " + std::to_string(rows) + " rows of " +
              std::to_string(tensor.dims[0]));
  }
  check(layer.repacks.size() == quantized, "planner repacks only the layer's quantized tensors");
}

// The rotary embedding and norms the kernels compute: a GGUF declares their
// RoPE base, rotated dimensions and RMS epsilon, and no RoPE scaling.
void checkRotary(const std::filesystem::path &directory) {
  const SmallTarget target = smallTarget(false);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "rotary.gguf";
  const auto planned = [&](const std::vector<test_gguf::Key> &keys) {
    splash::test::writeFile(path, test_gguf::file(keys, target.tensors));
    return plan(path, g);
  };
  const std::string scaling = g.architecture() + std::string(".rope.scaling.type");
  std::vector<test_gguf::Key> keys = metadata(g);
  keys.push_back(test_gguf::stringKey(scaling, "none"));
  check(!planned(keys).error, "planner accepts a GGUF that declares no RoPE scaling");
  keys.back() = test_gguf::stringKey(scaling, "yarn");
  check(names(planned(keys), {"rope.scaling.type yarn (expected none)"}), "planner refuses RoPE scaling");

  model::gguf::TargetGeometry other = g;
  other.rotaryPairs = 64;
  other.rotaryTheta = 1e6F;
  keys = metadata(other);
  for (test_gguf::Key &key : keys)
    if (key.name.ends_with(".attention.layer_norm_rms_epsilon")) key = test_gguf::float32Key(key.name, 1e-5F);
  check(names(planned(keys), {"rope.dimension_count 128 (expected 64)", "rope.freq_base 1e+06 (expected 1e+07)",
                              "attention.layer_norm_rms_epsilon 1e-05 (expected 1e-06)"}),
        "planner names another rotated width, RoPE base and RMS epsilon");

  keys = metadata(g);
  std::erase_if(keys, [](const test_gguf::Key &key) {
    return key.name.find(".rope.") != std::string::npos || key.name.ends_with(".layer_norm_rms_epsilon");
  });
  check(names(planned(keys), {"rope.dimension_count missing", "rope.freq_base missing",
                              "attention.layer_norm_rms_epsilon missing"}),
        "planner requires the rotary and norm metadata");
}

// A dense target stored for rotated inputs, every rotated input one rotation
// block wide: the rotation must name exactly the tensors the planner repacks,
// so Q8_0 alpha/beta, whose segment reads the rotated input, and not F32 ones.
void checkRotation(const std::filesystem::path &directory) {
  model::gguf::TargetGeometry g;
  g.layers = 2;
  g.hiddenSize = GGUF_ROTATION_BLOCK;
  g.vocabularySize = 256;
  g.intermediateSize = GGUF_ROTATION_BLOCK;
  g.gdnKeyHeads = 4;
  g.gdnValueHeads = 16;
  g.gdnHeadDimension = 64;
  g.convolutionDimension = 1536; // q and k of 4 heads, v of 16
  g.attentionWidth = GGUF_ROTATION_BLOCK; // four query heads of 256
  g.attentionKvHeads = 1;
  g.attentionHeadDimension = 256;
  g.rotaryPairs = 32;
  g.rotaryTheta = 1e7F;
  g.fullAttentionPeriod = 2; // layer 1
  const auto path = directory / "rotated.gguf";
  // PQ2_0 projections and token table, alpha/beta of type `alphaBeta`, and a
  // rotation that names every quantized tensor but the table, alpha/beta only
  // when `namesAlphaBeta`.
  const auto planned = [&](uint32_t alphaBeta, bool namesAlphaBeta) {
    TensorTypes types{{"ssm_beta.weight", alphaBeta}, {"ssm_alpha.weight", alphaBeta}};
    for (const char *name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
                             "attn_gate.weight", "ssm_out.weight", "ffn_gate.weight", "ffn_up.weight",
                             "ffn_down.weight", "output.weight", "token_embd.weight"})
      types[name] = model::ggml::kPQ2_0;
    const std::vector<Tensor> tensors = targetTensors(g, types);
    std::vector<std::string> weights;
    for (const Tensor &tensor : tensors)
      if (gguf_format_of(tensor.type) != GGUF_FMT_COUNT && tensor.name != "token_embd.weight" &&
          (namesAlphaBeta || !(tensor.name.ends_with("_alpha.weight") || tensor.name.ends_with("_beta.weight"))))
        weights.push_back(tensor.name);
    std::vector<test_gguf::Key> keys = metadata(g);
    const auto key = [](const char *name) { return std::string("prism.hadamard.") + name; };
    keys.push_back(test_gguf::uint32Key(key("version"), 1));
    keys.push_back(test_gguf::uint32Key(key("block_size"), GGUF_ROTATION_BLOCK));
    keys.push_back(test_gguf::stringKey(key("transform"), "normalized-sylvester-walsh-hadamard"));
    keys.push_back(test_gguf::stringKey(key("axis"), "input-last-dimension"));
    keys.push_back(test_gguf::stringKey(key("sign_mode"), "explicit"));
    keys.push_back(test_gguf::boolKey(key("gdn_v_grouped"), true));
    keys.push_back(test_gguf::int32ArrayKey(key("sign_widths"), {GGUF_ROTATION_BLOCK}));
    keys.push_back(test_gguf::int32ArrayKey(key("sign_values"), std::vector<int32_t>(GGUF_ROTATION_BLOCK, 1)));
    keys.push_back(test_gguf::stringArrayKey(key("weight_names"), weights));
    keys.push_back(test_gguf::stringArrayKey(key("inverse_weight_names"), {"token_embd.weight"}));
    splash::test::writeFile(path, test_gguf::file(keys, tensors));
    return plan(path, g);
  };
  const Plan floats = planned(model::ggml::kF32, false);
  check(!floats.error, "planner plans a rotation of F32 alpha/beta" + (floats.error ? ": " + *floats.error : ""));
  const Plan q8 = planned(model::ggml::kQ8_0, true);
  check(!q8.error, "planner plans a rotation that names Q8_0 alpha/beta" + (q8.error ? ": " + *q8.error : ""));
  check(names(planned(model::ggml::kQ8_0, false), {"the rotation must name every quantized tensor"}),
        "planner refuses a rotation that leaves out Q8_0 alpha/beta");
}

} // namespace

int main() {
  @autoreleasepool {
    const splash::test::TemporaryDirectory directory("splash-gguf-planner");
    guarded("planner on the dense target", [&] { checkDense(directory.path()); });
    guarded("planner on the MoE target", [&] { checkMoe(directory.path()); });
    guarded("planner on the rotary metadata", [&] { checkRotary(directory.path()); });
    guarded("planner on a rotated target", [&] { checkRotation(directory.path()); });
    std::printf("%s (%d failures)\n", failures ? "GGUF planner tests FAILED" : "GGUF planner tests passed", failures);
    return failures ? 1 : 0;
  }
}
