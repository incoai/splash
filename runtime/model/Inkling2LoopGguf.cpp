// Inkling-2-Mini's GGUF plan (Inkling2LoopLayout's ggufTargetPlan): its
// metadata checked key by key (gguf::MetadataCheck), and its images, a layer
// image per weight layer, then the head's and the embedding's, which the
// shared planner lays out from the tensors each holds (gguf::planTargetImages).

#include "model/GgufImage.hpp"
#include "model/GgufTarget.hpp"
#include "model/Inkling2Loop.hpp"

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {
namespace {

// The inputs a loop layer's message gates take, in the order of its message
// queries, as its tensors name them (loop_{q,k,v,ffn}_ctrl_*).
constexpr std::array<std::string_view, 4> kGateInputs{"q", "k", "v", "ffn"};

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
// expert's keys and no gating function. Every mismatch is named at once.
void requireMetadata(const GgufFile &file, const Inkling2LoopLayout &layout) {
  gguf::MetadataCheck check(file, Inkling2LoopLayout::ggufArchitecture);
  const uint32_t layers = layout.weightLayers;
  // An array of a value per weight layer, each the layer's.
  const auto perLayer = [layers](auto value) {
    return [layers, value](std::span<const double> values) {
      bool matches = values.size() == layers;
      for (uint32_t layer = 0; matches && layer < layers; ++layer) matches = values[layer] == double(value(layer));
      return matches;
    };
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
  check.expectArray("attention.sliding_window_pattern", perLayer([&](uint32_t layer) { return layout.slides(layer); }));
  check.expect("attention.relative_length", layout.relativeRank);
  check.expect("attention.relative_extent", layout.relativeExtent);
  check.expectFloat("attention.relative_clip", layout.relativeClip);
  check.expectFloat("attention.qk_clip", layout.queryClip);
  check.expect("attention.log_scale_floor", layout.logScalingFloor);
  check.expectFloat("attention.log_scale_alpha", layout.logScalingAlpha);
  check.expect("shortconv.l_cache", layout.convolutionTaps);
  check.expect("attn_res.block_size", layout.blockResidualPeriod);
  check.expect("loop.start", layout.loopFirstLayer);
  check.expect("loop.end", layout.loopEndLayer);
  check.expect("loop.count", layout.loopPasses);
  check.expect("loop.gate_rank", layout.loopGateRank);
  check.expect("loop.varying", 0);
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
// gate and relative scores (Q8_0 in its GGUF) and the sink projection (BF16),
// which the loader packs into one projection of the q input; the keys and
// values; the queries' and relative scores' learned scales (attn_q_clip,
// attn_rel_clip); the relative bias; the key and value convolutions; the
// output projection and its convolution (attn_post_conv); the FFN input's
// block mix query and norm, the router and routed experts (the fused gate
// and up, ffn_gate_up_exps, each NVFP4 with an expert's F32 .scale in layers
// 0-19), the router's bias (exp_probs_b) and the FFN output's convolution;
// and in a loop layer its message queries and its q, k, v and FFN gates,
// F32. The small F32 tensors as stored.
std::vector<gguf::Section> layerImage(const Inkling2LoopLayout &layout, uint32_t index) {
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
  add(Kind::SmallProjection, "attn_sink_proj.weight", layout.attentionQueryHeads, hidden);
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
  add(Kind::Values, "loop_msg_query.weight", kGateInputs.size() * hidden);
  for (const std::string_view input : kGateInputs) {
    const std::string stem = "loop_" + std::string(input) + "_ctrl_";
    add(Kind::FloatTensor, stem + "down.weight", layout.loopGateRank, 2 * hidden);
    add(Kind::FloatTensor, stem + "forget.weight", hidden, layout.loopGateRank);
    add(Kind::FloatTensor, stem + "gain.weight", hidden, layout.loopGateRank);
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

GgufTargetPlan ggufTargetPlan(const Inkling2LoopLayout &layout) {
  return {.layerImages = layout.weightLayers,
          .firstMoeLayer = 0,
          .images = [layout](const GgufFile &file, gguf::PlanOptions options) {
            requireMetadata(file, layout);
            return gguf::planTargetImages(
                file, layout, layout.weightLayers, options,
                [&](uint32_t index) { return layerImage(layout, index); }, headImage(layout));
          }};
}

} // namespace splash::model
