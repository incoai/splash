// The GGUF image planner on the small dense and MoE targets, a rotated dense
// one and a small Flash-Next one: where each tensor goes and in which row
// order, which F32 tensors it narrows to bf16, which float projections it
// quantizes to Q8_0, where a streamed table's rows are, which part of a split
// GGUF each tensor is read from, and the sources it refuses.
//   gguf-planner
#include "GgufFixtures.hpp"
#include "metal/abi/Gguf.h"
#include "model/GgufPreparation.hpp"
#include "model/GgufTarget.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/WeightStore.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <bit>
#include <memory>
#include <optional>

using namespace gguf_fixtures;

namespace {

// Q8_1, a type the loader does not read.
constexpr uint32_t kQ8_1 = ggmlType("Q8_1");

// The images of the GGUF at path, planned for geometry; the planner's error
// when it refuses the file.
struct Plan {
  std::vector<model::gguf::Image> images;
  std::optional<std::string> error;
};

Plan plan(const std::filesystem::path &path, const model::QwenTargetDimensions &geometry,
          model::gguf::PlanOptions options = {}) {
  try {
    model::WeightSource source(path);
    const model::GgufFile gguf(source);
    return {model::gguf::planImages(gguf, architecture(geometry), geometry, options), std::nullopt};
  } catch (const model::GgufError &error) {
    return {{}, error.what()};
  }
}

// The same repack: format, shape and sources' rows.
bool sameRepack(const model::gguf::Repack &left, const model::gguf::Repack &right) {
  if (left.format != right.format || left.rows != right.rows || left.columns != right.columns ||
      left.sources.size() != right.sources.size())
    return false;
  for (size_t index = 0; index < left.sources.size(); ++index) {
    const model::gguf::TensorRows &a = left.sources[index], &b = right.sources[index];
    if (a.name != b.name || a.offset != b.offset || a.first != b.first || a.rows != b.rows || a.rowBytes != b.rowBytes)
      return false;
  }
  return true;
}

// With streamed experts each layer's routed experts are the repacks the
// resident plan writes, out of the image, which keeps every other section and
// declares the resident image's bytes.
bool streamsExperts(const Plan &resident, const Plan &streamed) {
  if (resident.error || streamed.error || resident.images.size() != streamed.images.size()) return false;
  for (size_t index = 0; index < resident.images.size(); ++index) {
    const model::gguf::Image &whole = resident.images[index], &image = streamed.images[index];
    // Either declares the resident image's bytes (the weights' manifest).
    if (whole.residentBytes != whole.bytes || image.residentBytes != whole.bytes) return false;
    std::vector<const model::gguf::Repack *> routed, rest;
    for (const model::gguf::Repack &repack : whole.repacks)
      (repack.sources.front().name.find("_exps.") != std::string::npos ? routed : rest).push_back(&repack);
    if (routed.empty()) {
      if (image.experts || image.bytes != whole.bytes) return false;
      continue;
    }
    if (!image.experts || routed.size() != 3 || image.repacks.size() != rest.size() || image.bytes >= whole.bytes)
      return false;
    for (size_t projection = 0; projection < 3; ++projection)
      if (!sameRepack(*routed[projection], image.experts->projections[projection])) return false;
    for (size_t other = 0; other < rest.size(); ++other)
      if (!sameRepack(*rest[other], image.repacks[other])) return false;
  }
  return true;
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
             const model::QwenTargetDimensions &g) {
  return order.from == from && order.headRows == headRows && order.keyHeads == g.gdnKeyHeads &&
         order.valueHeadsPerKey == g.gdnValueHeads / g.gdnKeyHeads;
}

// A copy of the tensor's rows as the GGUF stores them.
bool asStored(const model::gguf::Copy *copy, const test_gguf::Bytes &data) {
  return copy && copy->conversion == model::gguf::Conversion::None && copy->source.order.from == UINT64_MAX &&
         copy->source.rows * copy->source.rowBytes == data.size();
}

void checkDense(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(false);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "dense.gguf";
  writeGguf(path, target.tensors, g);
  const Plan dense = plan(path, g);
  check(!dense.error, "planner plans the dense target" + (dense.error ? ": " + *dense.error : ""));
  if (dense.error) return;
  const model::gguf::Image &gdn = dense.images[0];

  for (const Tensor &tensor : target.tensors)
    if (isNorm(tensor.name))
      check(asStored(copyIn(dense, tensor.name), tensor.data), "planner keeps F32 values as stored: " + tensor.name);

  const uint32_t keyRows = g.convolutionDimension - g.gdnValueHeads * g.gdnHeadDimension;
  const auto *convolution = copyOf(gdn, "blk.0.ssm_conv1d.weight");
  check(convolution && convolution->conversion == model::gguf::Conversion::NarrowToBfloat16 &&
            convolution->source.rows == g.convolutionDimension &&
            grouped(convolution->source.order, keyRows, g.gdnHeadDimension, g),
        "planner narrows the convolution to bf16 and groups its value heads");
  const auto *decay = copyOf(gdn, "blk.0.ssm_a");
  check(decay && decay->conversion == model::gguf::Conversion::None && grouped(decay->source.order, 0, 1, g),
        "planner keeps the decay F32 and groups its value heads");
  const auto *bias = copyOf(gdn, "blk.0.ssm_dt.bias");
  check(bias && bias->conversion == model::gguf::Conversion::NarrowToBfloat16 && grouped(bias->source.order, 0, 1, g),
        "planner narrows the time bias to bf16 and groups its value heads");

  // A token table of every type the gathers read, F16 and BF16 among them, but llama.cpp's NVFP4, whose rows the
  // writer converts: copied as stored.
  std::vector<uint32_t> tableTypes;
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format)
    if (const std::optional<uint32_t> type = ggufType(format);
        type && gguf_embedding_format(format) && *type != model::ggml::kNVFP4)
      tableTypes.push_back(*type);
  for (const uint32_t type : tableTypes) {
    std::vector<Tensor> tensors = target.tensors;
    Tensor &table = tensorNamed(tensors, "token_embd.weight");
    const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(type);
    table.type = type;
    table.data.resize(uint64_t{g.vocabularySize} * g.hiddenSize / traits.blockElements * traits.blockBytes);
    writeGguf(path, tensors, g);
    const Plan tables = plan(path, g);
    check(!tables.error && asStored(copyIn(tables, "token_embd.weight"), table.data),
          "planner copies a token table as stored: " + model::ggmlTypeName(type) +
              (tables.error ? ": " + *tables.error : ""));
  }

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

// alpha/beta of any one GGUF tensor type of a format but BF16 and F16
// (GGUF's NVFP4 as NVFP4), on either architecture: one plane tile of rows of
// that format, the beta then the alpha rows at its native row bytes in grouped
// head order; a BF16 pair, one BF16 tensor of the beta then the alpha rows as
// stored; a pair of two types or of F16, one F32 tensor of the beta then the
// alpha rows, each decoded unless F32; a type the loader does not read,
// refused by name.
void checkQuantizedAlphaBeta(const std::filesystem::path &directory) {
  for (bool moe : {false, true}) {
    const SmallTarget target = smallTarget(moe);
    const model::QwenTargetDimensions &g = target.geometry;
    const auto path = directory / "alpha-beta.gguf";
    const auto tensorsWith = [&](uint32_t betaType, uint32_t alphaType) {
      std::vector<Tensor> tensors = target.tensors;
      for (const char *name : {"blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"}) {
        Tensor &tensor = tensorNamed(tensors, name);
        tensor.type = tensor.name.ends_with("_beta.weight") ? betaType : alphaType;
        const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(tensor.type);
        tensor.data.resize(uint64_t{g.gdnValueHeads} * g.hiddenSize / traits.blockElements * traits.blockBytes);
      }
      return tensors;
    };
    for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format) {
      const std::optional<uint32_t> stored = ggufType(format);
      if (!stored || format == GGUF_FMT_BF16 || format == GGUF_FMT_F16) continue;
      const uint32_t type = *stored;
      writeGguf(path, tensorsWith(type, type), g);
      const Plan result = plan(path, g);
      const auto *pair = result.error ? nullptr : repackOf(result.images[0], "blk.0.ssm_beta.weight");
      // Rows of the format's native blocks (llama.cpp's NVFP4 as NVFP4's).
      const auto heads = [&](const model::gguf::TensorRows &rows, const char *name) {
        return rows.name == name && rows.type == kQuantFormats[format].ggml_type && rows.rows == g.gdnValueHeads &&
               rows.rowBytes == gguf_reference::rowBytes(Fmt(format), g.hiddenSize) && grouped(rows.order, 0, 1, g);
      };
      check(pair && pair->format == format && pair->rows == QUANT_TILE_ROWS && pair->columns == g.hiddenSize &&
                pair->sources.size() == 2 && heads(pair->sources[0], "blk.0.ssm_beta.weight") &&
                heads(pair->sources[1], "blk.0.ssm_alpha.weight"),
            "planner alpha/beta: one plane tile of " + model::ggmlTypeName(type) + " beta then alpha rows (" +
                architecture(g) + ")" + (result.error ? ": " + *result.error : ""));
    }
    using model::gguf::Conversion;
    for (const auto &[beta, alpha] : {std::pair{kIQ4_XS, kQ8_0}, std::pair{kQ4_K, model::ggml::kF32},
                                      std::pair{model::ggml::kBF16, model::ggml::kNVFP4},
                                      std::pair{model::ggml::kF32, model::ggml::kBF16},
                                      std::pair{model::ggml::kF16, kQ8_0},
                                      std::pair{model::ggml::kBF16, model::ggml::kBF16},
                                      std::pair{model::ggml::kF16, model::ggml::kF16}}) {
      writeGguf(path, tensorsWith(beta, alpha), g);
      const Plan result = plan(path, g);
      const bool bfloat16 = beta == model::ggml::kBF16 && alpha == model::ggml::kBF16;
      const auto conversion = [&](uint32_t type) {
        return type == model::ggml::kF32 || bfloat16 ? Conversion::None : Conversion::DequantizeToFloat32;
      };
      const uint64_t rows = uint64_t{g.gdnValueHeads} * g.hiddenSize * (bfloat16 ? sizeof(uint16_t) : sizeof(float));
      const auto *b = copyIn(result, "blk.0.ssm_beta.weight"), *a = copyIn(result, "blk.0.ssm_alpha.weight");
      check(b && a && b->conversion == conversion(beta) && a->conversion == conversion(alpha) &&
                a->destination == b->destination + rows && grouped(b->source.order, 0, 1, g) &&
                grouped(a->source.order, 0, 1, g) && !repackOf(result.images[0], "blk.0.ssm_beta.weight"),
            std::string("planner makes alpha/beta one ") + (bfloat16 ? "BF16" : "F32") + " tensor: " +
                model::ggmlTypeName(beta) + " beta, " + model::ggmlTypeName(alpha) + " alpha" +
                (result.error ? ": " + *result.error : ""));
    }
    writeGguf(path, tensorsWith(kQ8_0, kQ8_1), g);
    check(names(plan(path, g), {"blk.0.ssm_alpha.weight (Q8_1)"}) &&
              !names(plan(path, g), {"blk.0.ssm_beta.weight"}),
          "planner refuses an alpha of a type it does not read by its name alone");
  }
}

void checkMoe(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(true);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "moe.gguf";

  model::QwenTargetDimensions wrongExperts = g;
  wrongExperts.experts = 5;
  writeGguf(path, target.tensors, wrongExperts);
  check(names(plan(path, g), {"expert_count"}), "planner names a metadata mismatch");
  model::QwenTargetDimensions dense = g;
  dense.ffnKind = model::QwenFfnKind::Dense;
  dense.intermediateSize = g.expertIntermediateSize;
  writeGguf(path, target.tensors, dense);
  check(names(plan(path, g), {"architecture", architecture(dense), architecture(g)}),
        "planner checks the architecture against the model's target");

  writeGguf(path, target.tensors, g);
  const Plan moe = plan(path, g);
  check(!moe.error, "planner plans the MoE target" + (moe.error ? ": " + *moe.error : ""));
  if (moe.error) return;
  const model::gguf::Image &layer = moe.images[0];
  const auto *beta = copyOf(layer, "blk.0.ssm_beta.weight"), *alpha = copyOf(layer, "blk.0.ssm_alpha.weight");
  check(beta && alpha && alpha->destination == beta->destination + target.data(beta->source.name).size() &&
            grouped(beta->source.order, 0, 1, g) && grouped(alpha->source.order, 0, 1, g) &&
            beta->conversion == model::gguf::Conversion::None && alpha->conversion == model::gguf::Conversion::None,
        "planner F32 alpha/beta tensor: beta then alpha rows in grouped order");
  for (const std::string name : {"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp_shexp.weight"})
    check(asStored(copyOf(layer, name), target.data(name)), "planner copies the F32 tensor as stored: " + name);

  // 16-bit floats as some GGUFs store them: a BF16 router as stored, which the
  // float projection runs; a BF16 shared-expert scalar gate and BF16 and F16
  // norms widened to the F32 values they hold, and a Q8_0 scalar gate decoded
  // to them. A router of another type is refused by name.
  {
    const auto sixteenPath = directory / "moe-16-bit-floats.gguf";
    std::vector<Tensor> sixteen = target.tensors;
    const std::pair<std::string, uint32_t> types[] = {{"blk.0.ffn_gate_inp.weight", model::ggml::kBF16},
                                                      {"blk.0.ffn_gate_inp_shexp.weight", model::ggml::kBF16},
                                                      {"blk.0.attn_norm.weight", model::ggml::kF16},
                                                      {"blk.0.post_attention_norm.weight", model::ggml::kBF16}};
    for (const auto &[name, type] : types) {
      Tensor &tensor = tensorNamed(sixteen, name);
      tensor.type = type;
      tensor.data.resize(tensor.data.size() / 2);
    }
    writeGguf(sixteenPath, sixteen, g);
    const Plan narrow = plan(sixteenPath, g);
    check(!narrow.error, "planner plans 16-bit float routers, gates and norms" +
                             (narrow.error ? ": " + *narrow.error : ""));
    if (!narrow.error) {
      const auto *router = copyOf(narrow.images[0], "blk.0.ffn_gate_inp.weight");
      check(asStored(router, tensorNamed(sixteen, "blk.0.ffn_gate_inp.weight").data) &&
                router->source.type == model::ggml::kBF16,
            "planner keeps a BF16 router as stored");
      for (const auto &[name, type] : types) {
        if (name == "blk.0.ffn_gate_inp.weight") continue;
        const auto *copy = copyOf(narrow.images[0], name);
        check(copy && copy->conversion == model::gguf::Conversion::WidenToFloat32,
              "planner widens the " + model::ggmlTypeName(type) + " " + name + " to F32");
      }
    }
    // A Q8_0 scalar gate, as quantized GGUFs store it: its native
    // rows, decoded to the F32 values they hold.
    Tensor &gate = tensorNamed(sixteen, "blk.0.ffn_gate_inp_shexp.weight");
    gate.type = kQ8_0;
    gate.data.resize(gguf_reference::rowBytes(gguf_reference::Q80, g.hiddenSize));
    writeGguf(sixteenPath, sixteen, g);
    const Plan decoded = plan(sixteenPath, g);
    const auto *decodedGate = decoded.error ? nullptr : copyOf(decoded.images[0], gate.name);
    check(decodedGate && decodedGate->conversion == model::gguf::Conversion::DequantizeToFloat32 &&
              decodedGate->source.rows == 1 && decodedGate->source.rowBytes == gate.data.size(),
          "planner decodes a Q8_0 shared-expert scalar gate to F32" + (decoded.error ? ": " + *decoded.error : ""));
    tensorNamed(sixteen, "blk.0.ffn_gate_inp.weight").type = model::ggml::kF16;
    writeGguf(sixteenPath, sixteen, g);
    check(names(plan(sixteenPath, g), {"blk.0.ffn_gate_inp.weight (F16)"}), "planner refuses an F16 router by name");

    // A BF16 router of 320 columns, not whole blocks of the 256 values BF16's
    // planes hold: rows of its values as stored all the same. The tensors of
    // hidden inputs are Q8_0, whose planes tile 64 columns.
    model::QwenTargetDimensions narrowHidden = g;
    narrowHidden.hiddenSize = 320;
    TensorTypes narrowTypes{{"ffn_gate_inp.weight", model::ggml::kBF16}};
    for (const char *name : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ffn_gate_exps.weight",
                             "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                             "ffn_up_shexp.weight", "ffn_down_shexp.weight", "output.weight", "token_embd.weight"})
      narrowTypes[name] = kQ8_0;
    std::vector<Tensor> narrowTensors = targetTensors(narrowHidden, narrowTypes);
    writeGguf(sixteenPath, narrowTensors, narrowHidden);
    const Plan narrowPlan = plan(sixteenPath, narrowHidden);
    const auto *narrowRouter =
        narrowPlan.error ? nullptr : copyOf(narrowPlan.images[0], "blk.0.ffn_gate_inp.weight");
    check(asStored(narrowRouter, tensorNamed(narrowTensors, "blk.0.ffn_gate_inp.weight").data) &&
              narrowRouter->source.rowBytes == narrowHidden.hiddenSize * sizeof(uint16_t),
          "planner keeps a BF16 router of 320 columns as stored" +
              (narrowPlan.error ? ": " + *narrowPlan.error : ""));
  }

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
  const Plan streamed = plan(path, g, {.streamExperts = true});
  check(streamsExperts(moe, streamed),
        "planner leaves streamed experts out of the layers' images, as the repacks it would write, and declares the "
        "images' bytes with them");
  // So the weights' manifest, the persistent cache's namespace, is the same with and without --expert-memory: the
  // target's loader records each image with its resident bytes (GgufTargetLoader::open).
  const auto manifest = [](const Plan &planned) {
    std::vector<model::WeightFileRecord> records;
    for (const model::gguf::Image &image : planned.images)
      records.push_back({"target/" + image.name, image.magic, image.layer, image.type, image.residentBytes, "sources"});
    return model::weightManifestFingerprint(records);
  };
  check(manifest(moe) == manifest(streamed), "planner gives the weights the same manifest whether the experts stream or not");

  // llama.cpp's fused ffn_gate_up_exps: the gate tensor takes each expert's first width rows of it, the up tensor
  // the next width; one of a type the loader does not read is refused by name.
  const uint64_t width = g.expertIntermediateSize;
  std::vector<Tensor> fused = withFusedGateUp(target.tensors, 0, g.experts);
  writeGguf(path, fused, g);
  const Plan fusedPlan = plan(path, g);
  std::vector<const model::gguf::Repack *> halves;
  if (!fusedPlan.error)
    for (const model::gguf::Repack &repack : fusedPlan.images[0].repacks)
      if (repack.sources.front().name == "blk.0.ffn_gate_up_exps.weight") halves.push_back(&repack);
  bool split = halves.size() == 2;
  for (size_t half = 0; split && half < 2; ++half) {
    split = halves[half]->rows == g.experts * width && halves[half]->sources.size() == g.experts;
    for (uint64_t expert = 0; split && expert < g.experts; ++expert) {
      const model::gguf::TensorRows &rows = halves[half]->sources[expert];
      split = rows.first == (2 * expert + half) * width && rows.rows == width &&
              rows.rowBytes == gguf_reference::rowBytes(Fmt(gguf_format_of(rows.type)), g.hiddenSize);
    }
  }
  check(split, "planner splits llama.cpp's fused gate/up experts into each expert's gate and up rows" +
                   (fusedPlan.error ? ": " + *fusedPlan.error : ""));
  check(streamsExperts(fusedPlan, plan(path, g, {.streamExperts = true})),
        "planner streams fused gate/up experts as each expert's gate and up rows");
  tensorNamed(fused, "blk.0.ffn_gate_up_exps.weight").type = kQ8_1;
  tensorNamed(fused, "blk.0.ffn_gate_up_exps.weight").data.resize(2 * g.experts * width * g.hiddenSize / 32 * 36);
  writeGguf(path, fused, g);
  check(names(plan(path, g), {"blk.0.ffn_gate_up_exps.weight (Q8_1)"}),
        "planner refuses a fused gate/up experts tensor of a type it does not read by name");
}

// The rotary embedding and norms the kernels compute: a GGUF declares their
// RoPE base, rotated dimensions and RMS epsilon, and no RoPE scaling.
void checkRotary(const std::filesystem::path &directory) {
  const SmallTarget target = smallTarget(false);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "rotary.gguf";
  const auto planned = [&](const std::vector<test_gguf::Key> &keys) {
    splash::test::writeFile(path, test_gguf::file(keys, target.tensors));
    return plan(path, g);
  };
  const std::string scaling = architecture(g) + ".rope.scaling.type";
  std::vector<test_gguf::Key> keys = metadata(g);
  keys.push_back(test_gguf::stringKey(scaling, "none"));
  check(!planned(keys).error, "planner accepts a GGUF that declares no RoPE scaling");
  keys.back() = test_gguf::stringKey(scaling, "yarn");
  check(names(planned(keys), {"rope.scaling.type yarn (expected none)"}), "planner refuses RoPE scaling");

  model::QwenTargetDimensions other = g;
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
// so quantized alpha/beta (Q8_0 and IQ4_XS here), whose segment reads the
// rotated input, and not F32 ones, nor BF16 (Prism's) or F16 ones, which load
// as F32.
void checkRotation(const std::filesystem::path &directory) {
  model::QwenTargetDimensions g;
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
  for (uint32_t type : {model::ggml::kF32, model::ggml::kBF16, model::ggml::kF16}) {
    const Plan floats = planned(type, false);
    check(!floats.error, "planner plans a rotation of " + model::ggmlTypeName(type) + " alpha/beta" +
                             (floats.error ? ": " + *floats.error : ""));
  }
  for (uint32_t type : {kQ8_0, kIQ4_XS}) {
    const std::string gates = model::ggmlTypeName(type) + " alpha/beta";
    const Plan named = planned(type, true);
    check(!named.error, "planner plans a rotation that names " + gates + (named.error ? ": " + *named.error : ""));
    check(names(planned(type, false), {"the rotation must name every quantized tensor"}),
          "planner refuses a rotation that leaves out " + gates);
  }
}

// The rows of a table that streams (gguf::tableRows), Flash-Next's PLE table
// (--ple-memory): where the file holds them, as its image copies them;
// refused for a type its image does not copy as stored and for another
// width.
void checkPleTableRows(const std::filesystem::path &directory) {
  const uint32_t columns = model::Qwen4ExpLayout{}.pleHeadDimension;
  std::mt19937 rng(7);
  const auto planned = [&](uint32_t type, uint64_t tableColumns, uint64_t rows, uint64_t rowBytes,
                           std::optional<model::gguf::StreamedRows> &out, test_gguf::Bytes &table) {
    table.resize(rows * rowBytes);
    for (uint8_t &byte : table) byte = uint8_t(rng());
    const auto path = directory / "ple.gguf";
    // A tensor before the table, so that it starts past the data's start.
    test_gguf::Bytes before(4 * 90);
    splash::test::writeFile(path,
                            test_gguf::file({}, {{"token_embd.weight", {columns, 4}, kIQ4_NL, before},
                                                 {"per_layer_token_embd.weight", {tableColumns, rows}, type, table}}));
    model::WeightSource source(path);
    const model::GgufFile gguf(source);
    try {
      out = model::gguf::tableRows(gguf, "per_layer_token_embd.weight", columns);
      return std::string();
    } catch (const model::GgufError &error) {
      return std::string(error.what());
    }
  };
  std::optional<model::gguf::StreamedRows> rows;
  test_gguf::Bytes table;
  check(planned(kIQ4_NL, columns, 1000, 90, rows, table).empty() && rows && rows->rows == 1000 &&
            rows->rowBytes == 90 && rows->format == GGUF_FMT_IQ4NL && rows->file == 0 &&
            rows->tensor == "per_layer_token_embd.weight",
        "an IQ4_NL PLE table streams as 90-byte rows");
  if (rows) {
    test_gguf::Bytes read(table.size());
    const int file = open((directory / "ple.gguf").c_str(), O_RDONLY);
    const bool same = file >= 0 && pread(file, read.data(), read.size(), off_t(rows->offset)) == ssize_t(read.size()) &&
                      read == table;
    if (file >= 0) close(file);
    check(same, "the streamed PLE rows are where the file holds the table");
  }
  rows.reset();
  check(planned(kQ8_0, columns, 100, 170, rows, table).empty() && rows && rows->rowBytes == 170 &&
            rows->format == GGUF_FMT_Q80,
        "a Q8_0 PLE table streams as 170-byte rows");
  rows.reset();
  check(planned(0, columns, 10, uint64_t{columns} * 4, rows, table).find("does not stream") != std::string::npos,
        "an F32 PLE table does not stream");
  check(planned(kIQ4_NL, 128, 10, 72, rows, table).find("unexpected shape") != std::string::npos,
        "a PLE table of another width does not stream");
}

// The plan's repack of the named tensor in any image, or nullptr.
const model::gguf::Repack *repackIn(const Plan &result, const std::string &name) {
  for (const model::gguf::Image &image : result.images)
    if (const auto *repack = repackOf(image, name)) return repack;
  return nullptr;
}

// Qwen3.8-Flash-Next's hyper-connection projections (GgufImage.cpp
// hyperImage) on the small qwen4exp target (qwen4ExpGeometry): of BF16 or
// F16, Q8_0 planes whose rows the reader builds as the reference GGUF
// quantization of Q8_0 does from the values (gguf_reference::quantizeQ8_0,
// which gguf-reference pins to it); of Q8_0, the rows as stored; of F32,
// refused.
void checkHyperConnections(const std::filesystem::path &directory) {
  const model::QwenTargetDimensions g = qwen4ExpGeometry();
  const auto path = directory / "hyper.gguf";
  // The plan's rows read `source`, which the last plan opened.
  std::unique_ptr<model::WeightSource> source;
  const auto planned = [&](const TensorTypes &hyperTypes, std::vector<Tensor> &tensors) -> Plan {
    TensorTypes types = qwen4ExpTypes();
    types.insert(hyperTypes.begin(), hyperTypes.end());
    tensors = targetTensors(g, types);
    // Random native blocks and F32 values, and BF16 or F16 ones of q8_0Values.
    uint32_t seed = 1300;
    for (Tensor &tensor : tensors) {
      uint64_t count = 1;
      for (uint64_t dim : tensor.dims) count *= dim;
      const uint32_t format = gguf_format_of(tensor.type);
      tensor.data.clear();
      if (tensor.type == model::ggml::kBF16 || tensor.type == model::ggml::kF16)
        for (float value : q8_0Values(count, ++seed)) {
          const uint16_t bits =
              tensor.type == model::ggml::kF16 ? gguf_reference::f2h(value) : splash::test::floatToBf16(value);
          tensor.data.insert(tensor.data.end(), {uint8_t(bits), uint8_t(bits >> 8)});
        }
      else if (format == GGUF_FMT_COUNT) tensor.data = floatValues(count, ++seed, readAsBfloat16(tensor));
      else tensor.data = fixture(Fmt(format), uint32_t(count / tensor.dims[0]), uint32_t(tensor.dims[0]), ++seed);
    }
    source.reset();
    writeGguf(path, tensors, g);
    try {
      source = std::make_unique<model::WeightSource>(path);
      const model::GgufFile gguf(*source);
      return {model::gguf::planImages(gguf, architecture(g), g), std::nullopt};
    } catch (const model::GgufError &error) {
      return {{}, error.what()};
    }
  };
  std::vector<Tensor> tensors;
  const Plan mixed = planned({{"hc_attn_down.weight", model::ggml::kBF16}, {"hc_attn_up.weight", model::ggml::kBF16},
                              {"hc_ffn_down.weight", model::ggml::kF16}, {"hc_ffn_up.weight", model::ggml::kF16},
                              {"output_hc_down.weight", kQ8_0}, {"output_hc_up.weight", kQ8_0}},
                             tensors);
  check(!mixed.error, "planner plans a target of BF16, F16 and Q8_0 hyper-connections" +
                          (mixed.error ? ": " + *mixed.error : ""));
  if (mixed.error) return;
  std::vector<std::string> projections{"output_hc_down.weight", "output_hc_up.weight"};
  for (uint32_t layer = 0; layer < g.layers; ++layer)
    for (const char *stem : {"hc_attn_down", "hc_attn_up", "hc_ffn_down", "hc_ffn_up"})
      projections.push_back("blk." + std::to_string(layer) + "." + stem + ".weight");
  for (const std::string &name : projections) {
    const Tensor &tensor = tensorNamed(tensors, name);
    const uint64_t rows = tensor.dims[1], columns = tensor.dims[0];
    const model::gguf::Repack *repack = repackIn(mixed, name);
    const bool stored = tensor.type == kQ8_0;
    // The Q8_0 rows of the values the tensor holds, or its rows as stored.
    std::vector<uint8_t> expected = tensor.data;
    if (!stored) {
      std::vector<float> values(rows * columns);
      for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t bits = uint16_t(tensor.data[2 * i] | tensor.data[2 * i + 1] << 8);
        values[i] = tensor.type == model::ggml::kF16 ? gguf_reference::h2f(bits)
                                                     : std::bit_cast<float>(uint32_t(bits) << 16);
      }
      expected = gguf_reference::quantizeQ8_0(values);
    }
    std::vector<uint8_t> rowsRead(expected.size());
    const bool shaped = repack && repack->format == GGUF_FMT_Q80 && repack->rows == rows &&
                        repack->columns == columns && repack->sources.size() == 1 &&
                        repack->sources[0].type == kQ8_0 &&
                        repack->sources[0].rowBytes == gguf_reference::rowBytes(Fmt(GGUF_FMT_Q80), uint32_t(columns)) &&
                        (stored ? !repack->sources[0].ggufFloatRowBytes
                                : repack->sources[0].ggufFloatType == tensor.type &&
                                      repack->sources[0].ggufFloatRowBytes == columns * 2);
    if (shaped) model::readRepackRows(*repack, 0, rows, rowsRead.data());
    check(shaped && rowsRead == expected, "planner repacks " + model::ggmlTypeName(tensor.type) + " " + name +
                                              (stored ? " as stored" : " as the reference Q8_0 of its values"));
  }
  check(names(planned({{"hc_ffn_up.weight", model::ggml::kF32}}, tensors),
              {"blk.0.hc_ffn_up.weight (F32)", "blk.1.hc_ffn_up.weight (F32)"}),
        "planner refuses an F32 hyper-connection projection by name");
}

// Qwen3.8-Flash-Next's PLE embedding on the small qwen4exp target
// (qwen4ExpGeometry): its projections, norms and convolution in the layer it
// feeds, and its table's rows as stored in an image of their own after the
// embedding's. With the table streamed, that image loads none of its bytes
// but stays planned, for the weights' records, its rows where the file holds
// them, and every other image is the same. A table that does not stream,
// GGUF's NVFP4, which the writer rebuilds, is refused by name where it
// would, and loads in its image, which the weights' sizes fall back on
// (model::streamableWeights). A GGUF that declares another PLE layer is
// refused.
void checkPleTable(const std::filesystem::path &directory) {
  const model::QwenTargetDimensions g = qwen4ExpGeometry();
  TensorTypes types = qwen4ExpTypes();
  for (const char *name : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
                           "output_hc_down.weight", "output_hc_up.weight"})
    types[name] = kQ8_0;
  std::vector<Tensor> tensors = targetTensors(g, types);
  const auto path = directory / "ple.gguf";
  writeGguf(path, tensors, g);
  const Plan loaded = plan(path, g), streamed = plan(path, g, {.streamTable = true});
  for (const Plan *result : {&loaded, &streamed})
    check(!result->error, "planner plans the PLE embedding" + (result->error ? ": " + *result->error : ""));
  if (loaded.error || streamed.error) return;
  const std::string p = "blk." + std::to_string(g.pleLayer) + ".";
  const model::gguf::Image &layer = loaded.images[g.pleLayer];
  check(repackOf(layer, p + "ple_key.weight") && repackOf(layer, p + "ple_value.weight") &&
            copyOf(layer, p + "ple_norm_key.weight") && copyOf(layer, p + "ple_norm_query.weight") &&
            copyOf(layer, p + "ple_norm_conv.weight") && copyOf(layer, p + "ple_conv1d.weight"),
        "planner plans the PLE projections, norms and convolution in the layer they feed");
  const model::gguf::Image &table = loaded.images.back();
  check(loaded.images.size() == g.layers + 3 && table.name == "ple.bin" && !table.streamedRows &&
            model::gguf::loadedBytes(table) == table.bytes &&
            asStored(copyOf(table, "per_layer_token_embd.weight"),
                     tensorNamed(tensors, "per_layer_token_embd.weight").data),
        "planner copies the PLE table's rows as stored into an image after the embedding's");
  const model::gguf::Image &streamedTable = streamed.images.back();
  const std::optional<model::gguf::StreamedRows> &rows = streamedTable.streamedRows;
  bool same = streamed.images.size() == loaded.images.size();
  for (size_t index = 0; same && index + 1 < loaded.images.size(); ++index)
    same = !streamed.images[index].streamedRows && streamed.images[index].bytes == loaded.images[index].bytes;
  check(same && rows && rows->tensor == "per_layer_token_embd.weight" && rows->rows == kPleTableRows &&
            rows->rowBytes == gguf_reference::rowBytes(Fmt(GGUF_FMT_Q80), g.pleHeadDimension) &&
            rows->format == GGUF_FMT_Q80 &&
            model::gguf::streamedTableBytes(streamedTable) == kPleTableRows * rows->rowBytes &&
            model::gguf::loadedBytes(streamedTable) == 0 && streamedTable.bytes == table.bytes,
        "planner streams the PLE table's rows instead of loading its image, which its record keeps");
  // NVFP4 rows are whole blocks of 256 values.
  model::QwenTargetDimensions wide = g;
  wide.pleHeadDimension = 256;
  TensorTypes rebuilt = types;
  rebuilt["per_layer_token_embd.weight"] = model::ggml::kNVFP4;
  writeGguf(path, targetTensors(wide, rebuilt), wide);
  const Plan whole = plan(path, wide), refused = plan(path, wide, {.streamTable = true});
  check(!whole.error && whole.images.back().name == "ple.bin" && !whole.images.back().streamedRows &&
            model::gguf::loadedBytes(whole.images.back()) && refused.error &&
            refused.error->find("does not stream: per_layer_token_embd.weight") != std::string::npos,
        "planner loads an NVFP4 PLE table and refuses by name to stream it" +
            (whole.error ? ": " + *whole.error : std::string()));
  model::QwenTargetDimensions other = g;
  other.pleLayer = 1;
  splash::test::writeFile(path, test_gguf::file(metadata(other), tensors));
  check(names(plan(path, g), {"ple.layers"}), "planner refuses a GGUF that declares another PLE layer");
}

// The dense target split in two as GGUF's split files store it, half
// its tensors in each part, last part written first: findTargetGgufs takes
// the parts in split order, past a file that is not a GGUF, and refuses
// another GGUF beside them or parts of another count. The plan of the parts
// writes the whole file's images, each tensor read from the part that holds
// it, at its offset in that part.
void checkSplit(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(false);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto whole = directory / "whole.gguf";
  writeGguf(whole, target.tensors, g);
  const auto split = directory / "split";
  std::filesystem::create_directory(split);
  const std::vector<std::filesystem::path> parts{split / "target-00001-of-00002.gguf",
                                                 split / "target-00002-of-00002.gguf"};
  const auto half = target.tensors.begin() + target.tensors.size() / 2;
  const std::vector<Tensor> held[2]{{target.tensors.begin(), half}, {half, target.tensors.end()}};
  std::map<std::string, uint32_t> partOf;
  for (uint16_t number : {1, 0}) {
    std::vector<test_gguf::Key> keys = number ? std::vector<test_gguf::Key>{} : metadata(g);
    for (test_gguf::Key &key : test_gguf::splitKeys(number, 2, int32_t(target.tensors.size())))
      keys.push_back(std::move(key));
    splash::test::writeFile(parts[number], test_gguf::file(keys, held[number]));
    for (const Tensor &tensor : held[number]) partOf[tensor.name] = number;
  }
  splash::test::writeFile(split / "target-00002-of-00002.gguf.incomplete", "a download in progress");
  check(model::findTargetGgufs(split) == parts, "findTargetGgufs takes a split GGUF's parts in split order");
  const auto refused = [&] {
    try {
      (void)model::findTargetGgufs(split);
    } catch (const model::GgufError &error) {
      return std::string_view(error.what()).find("not the parts of one split GGUF") != std::string_view::npos;
    }
    return false;
  };
  std::filesystem::copy_file(whole, split / "whole.gguf");
  check(refused(), "findTargetGgufs refuses another GGUF beside a split GGUF's parts");
  std::filesystem::remove(split / "whole.gguf");
  std::filesystem::rename(parts[1], split / "target-00002-of-00003.gguf");
  check(refused(), "findTargetGgufs refuses the parts of a split GGUF of another count");
  std::filesystem::rename(split / "target-00002-of-00003.gguf", parts[1]);

  // Both plans read their files, which stay open.
  const model::OpenGguf wholeGguf({whole}), splitGguf(model::findTargetGgufs(split));
  const model::GgufFile &file = splitGguf.file();
  const std::vector<model::gguf::Image> reference = model::gguf::planImages(wholeGguf.file(), architecture(g), g),
                                        images = model::gguf::planImages(file, architecture(g), g);
  // Whether a source of the parts' plan reads the part that holds its
  // tensor, whose bytes are at the source's offset in that part.
  bool partRead[2]{};
  const auto fromItsPart = [&](const model::gguf::TensorRows &rows) {
    const uint32_t number = partOf.at(rows.name);
    const test_gguf::Bytes &data = target.data(rows.name);
    test_gguf::Bytes read(data.size());
    if (rows.file != &file.source(number)) return false;
    rows.file->readData(rows.offset, read);
    partRead[number] = true;
    return read == data;
  };
  bool same = images.size() == reference.size();
  for (size_t index = 0; same && index < images.size(); ++index) {
    const model::gguf::Image &image = images[index], &expected = reference[index];
    same = image.name == expected.name && image.bytes == expected.bytes &&
           image.fills.size() == expected.fills.size() && image.copies.size() == expected.copies.size() &&
           image.repacks.size() == expected.repacks.size();
    for (size_t fill = 0; same && fill < image.fills.size(); ++fill)
      same = image.fills[fill].offset == expected.fills[fill].offset &&
             image.fills[fill].bytes == expected.fills[fill].bytes;
    for (size_t copy = 0; same && copy < image.copies.size(); ++copy)
      same = image.copies[copy].destination == expected.copies[copy].destination &&
             image.copies[copy].source.name == expected.copies[copy].source.name &&
             fromItsPart(image.copies[copy].source);
    for (size_t at = 0; same && at < image.repacks.size(); ++at) {
      const model::gguf::Repack &repack = image.repacks[at], &wholeRepack = expected.repacks[at];
      uint64_t rows = 0;
      for (const model::gguf::TensorRows &source : repack.sources) rows += source.rows;
      same = repack.format == wholeRepack.format && repack.columns == wholeRepack.columns &&
             repack.plane0 == wholeRepack.plane0 && repack.sources.size() == wholeRepack.sources.size() &&
             std::all_of(repack.sources.begin(), repack.sources.end(), fromItsPart);
      if (!same) break;
      // The rows the GPU repacks, which each source reads from its file.
      std::vector<uint8_t> read(rows * model::ggufRowBytes(kQuantFormats[repack.format], repack.columns)),
          wholeRead(read.size());
      model::readRepackRows(repack, 0, rows, read.data());
      model::readRepackRows(wholeRepack, 0, rows, wholeRead.data());
      same = read == wholeRead;
    }
  }
  check(same && partRead[0] && partRead[1],
        "planner reads each tensor of a split GGUF from its part into the whole file's images");
}

} // namespace

int main() {
  @autoreleasepool {
    const splash::test::TemporaryDirectory directory("splash-gguf-planner");
    guarded("planner on the dense target", [&] { checkDense(directory.path()); });
    guarded("planner on the MoE target", [&] { checkMoe(directory.path()); });
    guarded("planner on quantized alpha/beta", [&] { checkQuantizedAlphaBeta(directory.path()); });
    guarded("planner on the rotary metadata", [&] { checkRotary(directory.path()); });
    guarded("planner on a rotated target", [&] { checkRotation(directory.path()); });
    guarded("planner on a streamed PLE table", [&] { checkPleTableRows(directory.path()); });
    guarded("planner on Flash-Next hyper-connections", [&] { checkHyperConnections(directory.path()); });
    guarded("planner on Flash-Next's PLE table", [&] { checkPleTable(directory.path()); });
    guarded("planner on a split GGUF", [&] { checkSplit(directory.path()); });
    std::printf("%s (%d failures)\n", failures ? "GGUF planner tests FAILED" : "GGUF planner tests passed", failures);
    return failures ? 1 : 0;
  }
}
