// Inkling-2-Mini's GGUF plan (Inkling2LoopLayout's ggufTargetPlan), of
// either layout of its GGUFs (GgufLayout): its metadata checked key by key
// (gguf::MetadataCheck), and its images, a layer image per weight layer, then
// the head's and the embedding's, which the shared planner lays out from the
// tensors each holds (gguf::planTargetImages).

#include "model/GgufImage.hpp"
#include "model/GgufTarget.hpp"
#include "model/Inkling2Loop.hpp"
#include "model/Inkling2LoopSections.hpp"

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {
namespace {

// The two layouts of Inkling 2's GGUFs, of the same tensors of its weight
// layers: llama.cpp's converter's (inkling2_loop,
// Inkling2LoopLayout::ggufArchitecture), and Unsloth's own (inkling2, the
// root files of unsloth/Inkling-2-Mini-GGUF), which states its layer counts
// and per-layer arrays per layer application (`perApplication`), its loop by
// other keys and no loop.varying, names a loop layer's message gates by the
// input they gate (`gates`, in the order of its message queries: the q, k, v
// and FFN inputs) and keeps every loop layer's message queries in rows of two
// tensors of them all (messageQueries).
struct GgufLayout final {
  std::string_view architecture;
  bool perApplication;
  std::string_view loopStart, loopEnd, loopPasses;
  std::array<std::string_view, 4> gates;
};
constexpr GgufLayout kConverterLayout{Inkling2LoopLayout::ggufArchitecture,
                                      false,
                                      "loop.start",
                                      "loop.end",
                                      "loop.count",
                                      {"loop_q_ctrl_", "loop_k_ctrl_", "loop_v_ctrl_", "loop_ffn_ctrl_"}};
constexpr GgufLayout kUnslothLayout{"inkling2",
                                    true,
                                    "loop.core_start",
                                    "loop.core_end",
                                    "num_loops",
                                    {"loop_attn_", "loop_k_", "loop_v_", "loop_mlp_"}};

// The layout of `file`, by its architecture; MetadataCheck refuses any other.
const GgufLayout &ggufLayout(const GgufFile &file) {
  return file.metadata().architecture() == kUnslothLayout.architecture ? kUnslothLayout : kConverterLayout;
}

// A loop layer's message queries, of its q, k, v and FFN inputs: the
// converter's loop_msg_query of the four; or, in Unsloth's layout, a q and an
// FFN row of loop_res_query and a k and a v row of loop_kv_res_query, two a
// loop layer in its order.
gguf::Section messageQueries(const Inkling2LoopLayout &layout, const GgufLayout &gguf, uint32_t index) {
  const uint64_t rows = 4 * uint64_t{layout.hiddenSize};
  const std::string name = "blk." + std::to_string(index) + ".loop_msg_query.weight";
  if (!gguf.perApplication) return {gguf::Section::Kind::Values, name, rows};
  const uint64_t q = 2 * uint64_t{index - layout.loopFirstLayer};
  return {gguf::Section::Kind::Values,
          name,
          rows,
          0,
          {{"loop_res_query.weight", q, 1},
           {"loop_kv_res_query.weight", q, 1},
           {"loop_kv_res_query.weight", q + 1, 1},
           {"loop_res_query.weight", q + 1, 1}}};
}

// llama.cpp's converter writes its header (inkling2_loop.*): the shared
// sizes, the attention's head counts as one per weight layer
// (attention.head_count and head_count_kv arrays), no RoPE
// (rope.dimension_count 0, no rope.freq_base), its context and its
// vocabulary's rows and tokens, the sliding heads' widths and window and the
// sliding layers (a bool per weight layer), the relative bias, the clips,
// the log scaling, the convolutions' taps, the block residual's period, the
// loop (its start, end, passes, gate rank, and the same passes for every
// token: loop.varying false), the softcaps, the experts and their routing's
// scale (expert_weights_scale), and the experts' width, which it also
// writes as feed_forward_length (every FFN is a MoE). It writes no shared
// expert's keys and no gating function. Unsloth's layout states the same
// under its own architecture but its layer count and arrays per application
// and its loop's keys (GgufLayout). Every mismatch is named at once.
void requireMetadata(const GgufFile &file, const Inkling2LoopLayout &layout, const GgufLayout &gguf) {
  gguf::MetadataCheck check(file, gguf.architecture);
  const uint32_t layers = gguf.perApplication ? layout.layers : layout.weightLayers;
  // An array of a value per layer (weight layer or application), each the layer's.
  const auto perLayer = [layers](auto value) {
    return [layers, value](std::span<const double> values) {
      bool matches = values.size() == layers;
      for (uint32_t layer = 0; matches && layer < layers; ++layer) matches = values[layer] == double(value(layer));
      return matches;
    };
  };
  // Whether a layer slides: a weight layer as it does, an application as its weight layer does.
  const auto slides = [&](uint32_t layer) {
    return gguf.perApplication ? !layout.schedule().full(layer) : layout.slides(layer);
  };
  check.expect("block_count", layers);
  check.expect("context_length", layout.modelContextTokens);
  check.expect("embedding_length", layout.hiddenSize);
  check.expect("vocab_size", layout.vocabularySize);
  check.expect("vocab_size_unpadded", layout.vocabularyTokens);
  check.expectArray("attention.head_count", perLayer([&](uint32_t) { return layout.attentionQueryHeads; }));
  check.expectArray("attention.head_count_kv", perLayer([&](uint32_t) { return layout.attentionKvHeads; }));
  for (const char *key : {"attention.key_length", "attention.value_length", "attention.key_length_swa",
                          "attention.value_length_swa"})
    check.expect(key, layout.attentionHeadDimension);
  check.expect("rope.dimension_count", 0);
  check.expectFloat("attention.layer_norm_rms_epsilon", SPLASH_RMS_EPSILON);
  check.expect("attention.sliding_window", layout.slidingWindow);
  check.expectArray("attention.sliding_window_pattern", perLayer(slides));
  check.expect("attention.relative_length", layout.relativeRank);
  check.expect("attention.relative_extent", layout.relativeExtent);
  check.expectFloat("attention.relative_clip", layout.relativeClip);
  check.expectFloat("attention.qk_clip", layout.queryClip);
  check.expect("attention.log_scale_floor", layout.logScalingFloor);
  check.expectFloat("attention.log_scale_alpha", layout.logScalingAlpha);
  check.expect("shortconv.l_cache", layout.convolutionTaps);
  check.expect("attn_res.block_size", layout.blockResidualPeriod);
  check.expect(gguf.loopStart, layout.loopFirstLayer);
  check.expect(gguf.loopEnd, layout.loopEndLayer);
  check.expect(gguf.loopPasses, layout.loopPasses);
  check.expect("loop.gate_rank", layout.loopGateRank);
  if (!gguf.perApplication) check.expect("loop.varying", 0);
  check.expectFloat("activation.situ_beta", layout.expertGateSoftcap);
  check.expectFloat("activation.situ_linear_beta", layout.expertUpSoftcap);
  check.expectFloat("final_logit_softcapping", layout.logitSoftcap);
  check.expect("expert_count", layout.experts);
  check.expect("expert_used_count", layout.expertsPerToken);
  check.expect("expert_feed_forward_length", layout.expertIntermediateSize);
  check.expect("feed_forward_length", layout.expertIntermediateSize);
  check.expectFloat("expert_weights_scale", layout.expertRoutedScale);
  check.finish();
}

// A weight layer's image, in the order its loader reads it
// (model/Inkling2Loop.cpp readLayer): the block mix queries of its mixer's
// input (attn_res_query: none in layer 0, three rows in a full layer); a
// sliding layer's norm (attn_norm), or a full layer's norms of its q, k and v
// inputs (attn_{q,k,v}_inp_norm) and the projections that add to its key's
// and value's taps (attn_{k,v}_conv_proj, BF16 in its GGUF); the queries,
// gate and relative scores (Q8_0 in its GGUF) and the sink projection (BF16,
// or a quantized one's F32 values: llama-quantize's files quantize it as the
// queries, and the projection's fused tiles take at most three segments),
// which the loader packs into one projection of the q input; the keys and
// values; the queries' and relative scores' learned scales (attn_q_clip,
// attn_rel_clip); the relative bias; the key and value convolutions; the
// output projection and its convolution (attn_post_conv); the FFN input's
// block mix query and norm, the router and routed experts (the fused gate
// and up, ffn_gate_up_exps, each NVFP4 with an expert's F32 .scale in layers
// 0-19), the router's bias (exp_probs_b) and the FFN output's convolution;
// and in a loop layer its message queries and its q, k, v and FFN gates
// (F32, or a quantized gate's F32 values), by the names of the file's layout.
// The small F32 tensors as stored.
// Whether a loop layer's call of message gates, its q, k and v gates or its
// FFN gate, mixes formats as its image would store them, BF16 as stored and
// any other as F32: the fused kernels take a call's gates in one format, so
// such a call's BF16 gates widen to F32, exactly (kaitchup's dondon GGUFs
// keep some BF16 beside Q8_0 ones).
bool mixedCall(const GgufFile &file, const std::string &layer, const GgufLayout &gguf, bool ffn) {
  bool bfloat16 = false, other = false;
  for (size_t gate = ffn ? 3 : 0; gate < (ffn ? 4 : 3); ++gate)
    for (const char *projection : {"down", "forget", "gain"})
      if (const GgufTensor *tensor = file.find(layer + std::string(gguf.gates[gate]) + projection + ".weight"))
        (tensor->type == ggml::kBF16 ? bfloat16 : other) = true;
  return bfloat16 && other;
}

std::vector<gguf::Section> layerImage(const Inkling2LoopLayout &layout, const GgufLayout &gguf, uint32_t index,
                                      const GgufFile *file = nullptr) {
  using Kind = gguf::Section::Kind;
  const std::string p = "blk." + std::to_string(index) + ".";
  const uint64_t hidden = layout.hiddenSize, taps = layout.convolutionTaps, kvRows = layout.kvWidth();
  std::vector<gguf::Section> sections;
  const auto add = [&](Kind kind, const std::string &name, uint64_t rows, uint64_t columns = 0) {
    sections.push_back({kind, p + name, rows, columns});
  };
  if (const uint32_t queries = layout.mixerQueries(index)) add(Kind::Values, "attn_res_query.weight", queries * hidden);
  if (layout.slides(index)) {
    add(Kind::Norm, "attn_norm.weight", hidden);
  } else {
    for (const char *input : {"q", "k", "v"})
      add(Kind::Norm, "attn_" + std::string(input) + "_inp_norm.weight", hidden);
    add(Kind::SmallProjection, "attn_k_conv_proj.weight", layout.convolutionTapsWidth(), hidden);
    add(Kind::SmallProjection, "attn_v_conv_proj.weight", layout.convolutionTapsWidth(), hidden);
  }
  add(Kind::Quantized, "attn_q.weight", layout.attentionWidth, hidden);
  add(Kind::Quantized, "attn_gate.weight", layout.attentionWidth, hidden);
  add(Kind::Quantized, "attn_rel.weight", layout.relativeWidth(), hidden);
  add(Kind::FloatTensor, "attn_sink_proj.weight", layout.attentionQueryHeads, hidden);
  add(Kind::Quantized, "attn_k.weight", kvRows, hidden);
  add(Kind::Quantized, "attn_v.weight", kvRows, hidden);
  add(Kind::Values, "attn_q_clip.weight", layout.attentionWidth);
  add(Kind::Values, "attn_rel_clip.weight", layout.relativeWidth());
  add(Kind::Values, "attn_rel_bias.weight", uint64_t{layout.relativeExtent} * layout.relativeRank);
  add(Kind::Values, "attn_k_conv.weight", kvRows * taps);
  add(Kind::Values, "attn_v_conv.weight", kvRows * taps);
  add(Kind::Quantized, "attn_output.weight", hidden, layout.attentionWidth);
  add(Kind::Values, "attn_post_conv.weight", hidden * taps);
  add(Kind::Values, "ffn_res_query.weight", hidden);
  add(Kind::Norm, "ffn_norm.weight", hidden);
  add(Kind::Router, "ffn_gate_inp.weight", layout.experts, hidden);
  // The prefix of the layer's expert tensors (Builder::routedExperts).
  sections.push_back({Kind::RoutedExperts, p, layout.experts, layout.expertIntermediateSize});
  add(Kind::Values, "exp_probs_b.bias", layout.experts);
  add(Kind::Values, "ffn_post_conv.weight", hidden * taps);
  if (!layout.schedule().isLoopLayer(index)) return sections;
  sections.push_back(messageQueries(layout, gguf, index));
  const bool mixer = file && mixedCall(*file, p, gguf, false), ffn = file && mixedCall(*file, p, gguf, true);
  for (size_t gate = 0; gate < gguf.gates.size(); ++gate) {
    const std::string stem(gguf.gates[gate]);
    const size_t first = sections.size();
    add(Kind::FloatTensor, stem + "down.weight", layout.loopGateRank, 2 * hidden);
    add(Kind::FloatTensor, stem + "forget.weight", hidden, layout.loopGateRank);
    add(Kind::FloatTensor, stem + "gain.weight", hidden, layout.loopGateRank);
    for (size_t at = first; at < sections.size(); ++at) sections[at].widened = gate == 3 ? ffn : mixer;
  }
  return sections;
}

// The head's image before the head itself (output.weight): the token
// table's norm, which the embedding's rows read after their gather, the
// head's block mix query and the final norm.
std::vector<gguf::Section> headImage(const Inkling2LoopLayout &layout) {
  using Kind = gguf::Section::Kind;
  return {{Kind::Norm, "token_embd_norm.weight", layout.hiddenSize},
          {Kind::Values, "output_res_query.weight", layout.hiddenSize},
          {Kind::Norm, "output_norm.weight", layout.hiddenSize}};
}

} // namespace

// The sections of its images by llama.cpp's converter's names, which its
// safetensors plan reads checkpoints by too.
std::vector<gguf::Section> inkling2::layerSections(const Inkling2LoopLayout &layout, uint32_t index) {
  return layerImage(layout, kConverterLayout, index);
}
std::vector<gguf::Section> inkling2::headSections(const Inkling2LoopLayout &layout) { return headImage(layout); }

GgufTargetPlan ggufTargetPlan(const Inkling2LoopLayout &layout) {
  return {.layerImages = layout.weightLayers,
          .firstMoeLayer = 0,
          .images = [layout](const GgufFile &file, gguf::PlanOptions options) {
            const GgufLayout &gguf = ggufLayout(file);
            requireMetadata(file, layout, gguf);
            return gguf::planTargetImages(
                file, layout, layout.weightLayers, options,
                [&](uint32_t index) { return layerImage(layout, gguf, index, &file); }, headImage(layout));
          }};
}

} // namespace splash::model
