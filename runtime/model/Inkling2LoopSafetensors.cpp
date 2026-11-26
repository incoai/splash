// Inkling-2-Mini's safetensors plan (Inkling2LoopLayout's
// safetensorsTargetPlan): the sections its GGUF plan lays out
// (model/Inkling2LoopSections.hpp), each read from a checkpoint by its names
// there, which the shared planner plans (safetensors::planTargetImages). The
// release's names (thinkingmachines/Inkling-2-Mini, model.llm.*) come first;
// its MLX converters' are aliases of them: oMLX's (model.*, a sliding layer's
// queries, keys, values, relative scores, gate and sinks fused into one
// in_proj, a full layer's queries, relative scores, gate and sinks into its
// q_proj and its convolution's projections into its k_proj and v_proj, its
// message gates packed a loop layer's in two tensors, transposed), mlx-vlm's
// (model.language_model.*, transformers' names) and mlx-lm's
// (language_model.model.*, its experts' gate and up apart). Each holds the
// release's tensors as llama.cpp's converter reads them into its GGUFs
// (conversion/inkling2.py), checked value for value against the release and
// its GGUF: the relative bias and a full layer's static convolution taps
// stored transposed (the taps last first), the experts' gate and up rows
// interleaved (but in mlx-lm's), and the release's message queries in two
// tensors of every loop layer's.

#include "model/Inkling2Loop.hpp"
#include "model/Inkling2LoopSections.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/WeightStore.hpp"

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {
namespace {

using safetensors::SectionSource;

// The names a checkpoint takes, told by its token table's: the release's,
// oMLX's, mlx-vlm's or mlx-lm's.
enum Names : uint8_t { Release, OMlx, MlxVlm, MlxLm, kNamings };
constexpr std::array<std::string_view, kNamings> kTables{"model.llm.embed", "model.embed",
                                                         "model.language_model.embed_tokens",
                                                         "language_model.model.embed_tokens"};
constexpr std::array<std::string_view, kNamings> kModels{"model.llm.", "model.", "model.language_model.",
                                                         "language_model.model."};
Names namesOf(const SafetensorsCheckpoint &checkpoint) {
  for (uint8_t names = 0; names < kNamings; ++names)
    if (checkpoint.find(std::string(kTables[names]) + ".weight")) return Names(names);
  throw WeightStoreError("the checkpoint holds no Inkling-2-Mini token table: model.llm.embed (its release's), "
                         "model.embed (oMLX's), model.language_model.embed_tokens (mlx-vlm's) or "
                         "language_model.model.embed_tokens (mlx-lm's)");
}

// A section of a weight layer's image and its source in each naming, after
// the layer's prefix: a tensor, or a module (a projection's, the router's).
struct Alias final {
  std::string_view section;
  std::array<std::string_view, kNamings> names;
};
constexpr Alias kLayerAliases[] = {
    {"attn_norm.weight", {"attn_norm.weight", "attn_norm.weight", "input_layernorm.weight", "input_layernorm.weight"}},
    {"attn_q_inp_norm.weight",
     {"q_input_norm.weight", "q_input_norm.weight", "q_input_layernorm.weight", "q_input_layernorm.weight"}},
    {"attn_k_inp_norm.weight",
     {"k_input_norm.weight", "k_input_norm.weight", "k_input_layernorm.weight", "k_input_layernorm.weight"}},
    {"attn_v_inp_norm.weight",
     {"v_input_norm.weight", "v_input_norm.weight", "v_input_layernorm.weight", "v_input_layernorm.weight"}},
    {"attn_q.weight", {"attn.wq_du", "", "self_attn.q_proj", "self_attn.q_proj"}},
    {"attn_gate.weight", {"attn.gate_attn", "", "self_attn.gate_proj", "self_attn.gate_proj"}},
    {"attn_rel.weight", {"attn.wr_du", "", "self_attn.r_proj", "self_attn.r_proj"}},
    {"attn_sink_proj.weight", {"attn.attn_sink_proj", "", "self_attn.sink_proj", "self_attn.sink_proj"}},
    {"attn_k.weight", {"attn.wk_dv", "", "self_attn.k_proj", "self_attn.k_proj"}},
    {"attn_v.weight", {"attn.wv_dv", "", "self_attn.v_proj", "self_attn.v_proj"}},
    {"attn_k_conv_proj.weight",
     {"attn.k_sconv.weight_proj", "", "self_attn.k_sconv.weight_proj", "self_attn.k_sconv.weight_proj"}},
    {"attn_v_conv_proj.weight",
     {"attn.v_sconv.weight_proj", "", "self_attn.v_sconv.weight_proj", "self_attn.v_sconv.weight_proj"}},
    {"attn_q_clip.weight", {"attn.q_l2_clip.scale", "attn.q_scale", "self_attn.q_l2_clip.scale", "self_attn.q_scale"}},
    {"attn_rel_clip.weight",
     {"attn.rel_l2_clip.scale", "attn.r_scale", "self_attn.rel_l2_clip.scale", "self_attn.rel_scale"}},
    {"attn_output.weight", {"attn.wo_ud", "attn.o_proj", "self_attn.o_proj", "self_attn.o_proj"}},
    {"attn_post_conv.weight",
     {"attn_sconv.weight", "attn_conv", "post_attention_convolution.conv1d.weight", "attn_sconv.conv.weight"}},
    {"ffn_res_query.weight", {"", "mlp_input.query", "mlp_input.query", "mlp_input.query"}},
    {"ffn_norm.weight",
     {"mlp_norm.weight", "mlp_norm.weight", "post_attention_layernorm.weight", "post_attention_layernorm.weight"}},
    {"ffn_gate_inp.weight", {"mlp.gate", "mlp.router_weight", "mlp.gate", "mlp.gate_weight"}},
    {"exp_probs_b.bias",
     {"mlp.gate.bias", "mlp.router_bias", "mlp.gate.e_score_correction_bias", "mlp.e_score_correction_bias"}},
    {"ffn_post_conv.weight",
     {"mlp_sconv.weight", "mlp_conv", "post_mlp_convolution.conv1d.weight", "mlp_sconv.conv.weight"}},
};

// oMLX's rows of its fused projections: a sliding layer's in_proj (queries,
// keys, values, relative scores, gate, sinks) and a full layer's q_proj
// (queries, relative scores, gate, sinks), k_proj and v_proj (keys or values,
// then their convolution's projection).
struct FusedRows final {
  std::string_view section, module;
  bool sliding;
  uint64_t first, rows;
};
FusedRows fusedRows(const Inkling2LoopLayout &layout, std::string_view section, bool sliding) {
  const uint64_t q = layout.attentionWidth, kv = layout.kvWidth(), rel = layout.relativeWidth(),
                 sinks = layout.attentionQueryHeads, taps = layout.convolutionTapsWidth();
  const FusedRows rows[] = {
      {"attn_q.weight", "attn.in_proj", true, 0, q},
      {"attn_k.weight", "attn.in_proj", true, q, kv},
      {"attn_v.weight", "attn.in_proj", true, q + kv, kv},
      {"attn_rel.weight", "attn.in_proj", true, q + 2 * kv, rel},
      {"attn_gate.weight", "attn.in_proj", true, q + 2 * kv + rel, q},
      {"attn_sink_proj.weight", "attn.in_proj", true, 2 * q + 2 * kv + rel, sinks},
      {"attn_q.weight", "attn.q_proj", false, 0, q},
      {"attn_rel.weight", "attn.q_proj", false, q, rel},
      {"attn_gate.weight", "attn.q_proj", false, q + rel, q},
      {"attn_sink_proj.weight", "attn.q_proj", false, 2 * q + rel, sinks},
      {"attn_k.weight", "attn.k_proj", false, 0, kv},
      {"attn_k_conv_proj.weight", "attn.k_proj", false, kv, taps},
      {"attn_v.weight", "attn.v_proj", false, 0, kv},
      {"attn_v_conv_proj.weight", "attn.v_proj", false, kv, taps},
  };
  for (const FusedRows &entry : rows)
    if (entry.section == section && entry.sliding == sliding) return entry;
  throw WeightStoreError("oMLX names no rows for " + std::string(section));
}

// The release's controllers and mlx-vlm's and mlx-lm's message gates by the
// gate's input (loop_{q,k,v,ffn}_ctrl_*), and the names of each one's
// projections (down, forget, gain).
constexpr std::array<std::string_view, 4> kGateInputs{"q", "k", "v", "ffn"};
constexpr std::array<std::string_view, 4> kControllers{"attn", "k", "v", "mlp"};
constexpr std::array<std::string_view, 3> kGateProjections{"down", "forget", "gain"};
constexpr std::array<std::string_view, 3> kMessageGateProjections{"down_proj", "forget_gate_proj",
                                                                  "message_gate_proj"};

// The sources of section `rest` of weight layer `layer` (blk.<layer>.<rest>)
// in `checkpoint`, of `names`.
std::vector<SectionSource> layerSources(const SafetensorsCheckpoint &checkpoint, Names names,
                                        const Inkling2LoopLayout &layout, uint32_t layer, std::string_view rest) {
  const std::string model(kModels[names]), p = model + "layers." + std::to_string(layer) + ".";
  const bool sliding = layout.slides(layer);
  const uint64_t hidden = layout.hiddenSize;
  // The release's block mix queries, of every layer in one list each: a
  // mixer's input's of layer i in attn_queries.(i - 1), an FFN's in
  // mlp_queries.i.
  if (rest == "attn_res_query.weight")
    return {{names == Release ? model + "attn_res.attn_queries." + std::to_string(layer - 1)
                              : p + (names == OMlx ? "attn_input.query" : "attention_input.query")}};
  if (rest == "ffn_res_query.weight" && names == Release)
    return {{model + "attn_res.mlp_queries." + std::to_string(layer)}};
  // A key's or value's convolution taps: a full layer's static taps,
  // transposed and last first.
  for (const std::string_view input : {"k", "v"}) {
    if (rest != "attn_" + std::string(input) + "_conv.weight") continue;
    const std::string sconv = (names == Release || names == OMlx ? "attn." : "self_attn.") + std::string(input);
    if (sliding)
      return {{p + (names == OMlx              ? sconv + "_conv"
                    : names == Release         ? sconv + "_sconv.weight"
                    : names == MlxVlm          ? sconv + "_sconv.conv1d.weight"
                                               : sconv + "_sconv.conv.weight")}};
    return {{.name = p + (names == OMlx ? sconv + "_static" : sconv + (names == Release ? "_sconv.static_w"
                                                                                          : "_sconv.static_weight")),
             .transposed = true,
             .reversed = true}};
  }
  if (rest == "attn_rel_bias.weight")
    return {{.name = p + (names == Release   ? "attn.rel_logits_proj.proj"
                          : names == OMlx    ? "attn.rel_proj"
                          : names == MlxVlm ? "self_attn.rel_logits_proj.proj"
                                             : "self_attn.rel_proj"),
             .transposed = true}};
  // The routed experts: gate and up the windows of the release's row pairs.
  if (rest.empty()) {
    if (names == MlxLm)
      return {{p + "mlp.switch_mlp.gate_proj"}, {p + "mlp.switch_mlp.up_proj"}, {p + "mlp.switch_mlp.down_proj"}};
    const std::string experts = p + "mlp.experts.";
    const std::string gateUp = experts + (names == MlxVlm ? "gate_up_proj" : "w13");
    return {{.name = gateUp, .windows = 2, .window = 0},
            {.name = gateUp, .windows = 2, .window = 1},
            {experts + (names == MlxVlm ? "down_proj" : "w2")}};
  }
  // A loop layer's message queries: the release's rows of its two lists,
  // each loop layer's two of each, its q and FFN queries and its k and v
  // queries; mlx-vlm's and mlx-lm's rows of one list.
  if (rest == "loop_msg_query.weight") {
    const uint64_t index = layer - layout.loopFirstLayer;
    if (names == Release) {
      const std::string queries = model + "middle_cycle_cross_pass_queries";
      return {{.name = queries, .first = 2 * index, .rows = 1},
              {.name = model + "middle_cycle_cross_pass_kv_queries", .first = 2 * index, .rows = 2},
              {.name = queries, .first = 2 * index + 1, .rows = 1}};
    }
    if (names == OMlx) return {{p + "loop_message.query"}};
    // mlx-lm's earlier conversions name the list message_queries.
    const std::string list = model + "message_input.query";
    return {{.name = checkpoint.find(list) ? list : model + "message_queries", .first = 4 * index, .rows = 4}};
  }
  // A loop layer's message gates: oMLX's packed and transposed, a gate's
  // down projection rows of its loop_down and its gain and forget the column
  // windows of its rows of loop_gain_forget; the others' tensors, by the
  // release's names (which mlx-lm's earlier conversions keep) or mlx-vlm's
  // and mlx-lm's. The fused kernels take a call's down projections in one
  // format and its forget and gain projections in one format
  // (ops::MessageGate): the mixer's q, k and v gates, or the FFN's
  // (kInkling2LoopGateCalls). A group whose projections mix BF16 and F32
  // takes its BF16 rows widened to F32, exactly; mlx-community's conversions
  // (BF16 down, F32 forget and gain) keep each group in its stored format,
  // saving 27.0 MiB on Inkling-2-Mini.
  const auto controller = [&](size_t gate, size_t projection) {
    return p + "controllers." + std::string(kControllers[gate]) + "." + std::string(kGateProjections[projection]) +
           "_weight";
  };
  const auto gateTensor = [&](size_t gate, size_t projection) {
    if (names == Release || checkpoint.find(controller(gate, projection))) return controller(gate, projection);
    return p + (gate == 3 ? std::string("mlp") : std::string(kGateInputs[gate])) + "_message_gate." +
           std::string(kMessageGateProjections[projection]) + ".weight";
  };
  const auto mixedGroup = [&](size_t gate, bool down) {
    bool bfloat16 = false, float32 = false;
    const Inkling2LoopGateCall call = inkling2LoopGateCall(gate);
    const size_t first = down ? 0 : 1, end = down ? 1 : kGateProjections.size();
    for (size_t other = call.first; other < call.end; ++other)
      for (size_t projection = first; projection < end; ++projection)
        if (const SourceTensor *tensor = checkpoint.find(gateTensor(other, projection))) {
          bfloat16 = bfloat16 || tensor->dtype == "BF16";
          float32 = float32 || tensor->dtype == "F32";
        }
    return bfloat16 && float32;
  };
  for (size_t gate = 0; gate < kGateInputs.size(); ++gate)
    for (size_t projection = 0; projection < kGateProjections.size(); ++projection) {
      if (rest != "loop_" + std::string(kGateInputs[gate]) + "_ctrl_" + std::string(kGateProjections[projection]) +
                      ".weight")
        continue;
      if (names == OMlx && !checkpoint.find(controller(gate, projection))) {
        const uint64_t rank = layout.loopGateRank;
        if (projection == 0)
          return {{.name = p + "loop_down", .first = gate * 2 * hidden, .rows = 2 * hidden, .transposed = true}};
        return {{.name = p + "loop_gain_forget",
                 .first = gate * rank,
                 .rows = rank,
                 .windows = 2,
                 .window = projection == 1 ? 1u : 0u,
                 .transposed = true}};
      }
      return {{.name = gateTensor(gate, projection), .widened = mixedGroup(gate, projection == 0)}};
    }
  // Every other section by its aliases, oMLX's projections as its fused rows.
  for (const Alias &alias : kLayerAliases) {
    if (alias.section != rest) continue;
    if (names == OMlx && alias.names[OMlx].empty()) {
      const FusedRows fused = fusedRows(layout, rest, sliding);
      return {{.name = p + std::string(fused.module), .first = fused.first, .rows = fused.rows}};
    }
    if (!alias.names[names].empty()) return {{p + std::string(alias.names[names])}};
  }
  throw WeightStoreError("no Inkling-2-Mini checkpoint tensor is known for blk." + std::to_string(layer) + "." +
                         std::string(rest));
}

// The sources of a head or embedding section.
std::vector<SectionSource> headSources(Names names, std::string_view section) {
  const std::string model(kModels[names]);
  const bool release = names == Release;
  if (section == "token_embd_norm.weight") return {{model + "embed_norm.weight"}};
  if (section == "output_res_query.weight")
    return {{model + (release ? "attn_res.output_query" : "final_input.query")}};
  if (section == "output_norm.weight") return {{model + "norm.weight"}};
  if (section == "output.weight")
    return {{release ? model + "unembed" : names == MlxLm ? std::string("language_model.lm_head") : "lm_head"}};
  if (section == "token_embd.weight") return {{std::string(kTables[names])}};
  throw WeightStoreError("no Inkling-2-Mini checkpoint tensor is known for " + std::string(section));
}

// The sources of section `name` in `checkpoint`, of `names`.
std::vector<SectionSource> sources(const SafetensorsCheckpoint &checkpoint, Names names,
                                   const Inkling2LoopLayout &layout, std::string_view name) {
  if (!name.starts_with("blk.")) return headSources(names, name);
  const size_t dot = name.find('.', 4);
  return layerSources(checkpoint, names, layout, uint32_t(std::stoul(std::string(name.substr(4, dot - 4)))),
                      name.substr(dot + 1));
}

} // namespace

SafetensorsTargetPlan safetensorsTargetPlan(const Inkling2LoopLayout &layout) {
  return {.layerImages = layout.weightLayers,
          .firstMoeLayer = 0,
          .images = [layout](const SafetensorsCheckpoint &checkpoint, gguf::PlanOptions options) {
            const Names names = namesOf(checkpoint);
            return SafetensorsTargetPlan::Images{
                safetensors::planTargetImages(
                    checkpoint, layout, layout.weightLayers, options,
                    [&](uint32_t index) { return inkling2::layerSections(layout, index); },
                    inkling2::headSections(layout),
                    [&](const gguf::Section &section) { return sources(checkpoint, names, layout, section.name); }),
                true};
          }};
}

} // namespace splash::model
