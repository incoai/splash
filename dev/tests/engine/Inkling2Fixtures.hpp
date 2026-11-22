#pragma once

// The GGUFs of small Inkling-2-Mini targets (model/Inkling2Loop.hpp), which
// the planner test and the family's engine test write: their tensors as
// llama.cpp's converter writes the model's, of seeded values in a
// checkpoint's sizes, their header's keys, and the engine test's layout.

#include "GgufFixtures.hpp"
#include "model/Inkling2Loop.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gguf_fixtures {

// The values of the Inkling fixture are a checkpoint's in size, so that its
// forward stays in a model's range (the engine's half tiles hold it): a
// projection's weights about 1/sqrt(inputs), as its training leaves them;
// a quantized block's stored scales set to that (Q8_0 its d, Q5_K its d and
// dmin, each of a block's typical |code| to a weight), each a little apart.
inline float projectionScale(uint64_t inputs) { return 1.0f / std::sqrt(float(inputs)); }
// A tensor's factor beside its projection's size: a layer's outputs onto the
// residual (attn_output, the experts' down) a quarter, so that each layer
// moves its rows a little, as a trained model's do, and rounding differences
// do not grow from one application to the next.
inline float outputFactor(const std::string &name) {
  return name.find("attn_output.") != std::string::npos || name.find("ffn_down_exps.") != std::string::npos ? 0.25f
                                                                                                          : 1.0f;
}
inline float loadFloat(const std::vector<uint8_t> &bytes, uint64_t i) {
  float value;
  std::memcpy(&value, bytes.data() + i * 4, 4);
  return value;
}
inline void storeFloat(std::vector<uint8_t> &bytes, uint64_t i, float value) {
  std::memcpy(bytes.data() + i * 4, &value, 4);
}
inline void storeBf16(std::vector<uint8_t> &bytes, uint64_t i, float value) {
  const uint16_t bits = splash::test::floatToBf16(value);
  std::memcpy(bytes.data() + i * 2, &bits, 2);
}
inline void scaleBlocks(Tensor &tensor, uint64_t inputs, std::mt19937 &rng) {
  std::uniform_real_distribution<float> spread(0.75f, 1.25f);
  const bool q8 = tensor.type == kQ8_0;
  // Q8_0: codes up to 127 (typical 73); Q5_K: 6-bit scales and mins by
  // 5-bit codes (typical 32 x 16).
  const uint64_t blockBytes = q8 ? 34 : 176;
  const float code = q8 ? 73.0f : 512.0f;
  if (!q8 && tensor.type != kQ5_K) throw std::logic_error("the Inkling fixture scales Q8_0 and Q5_K blocks");
  for (uint64_t at = 0; at < tensor.data.size(); at += blockBytes)
    for (uint64_t field = 0; field < (q8 ? 1u : 2u); ++field) {
      const uint16_t d = gguf_reference::f2h(spread(rng) * projectionScale(inputs) * outputFactor(tensor.name) / code);
      std::memcpy(tensor.data.data() + at + 2 * field, &d, 2);
    }
}
// An F32 tensor's factor: a message gate's projections 1/sqrt(inputs), the
// router a tenth of that (its scores near 0.5, which its bias ranks); a
// convolution's taps a quarter, so that a tap's sum stays about its row's;
// the queries' scale and the relative bias a tenth; norms, the relative
// scale and the mixes' queries as floatValues gives them, of size 0.5 to 1.
inline float inkling2FloatScale(const Tensor &tensor) {
  const std::string &name = tensor.name;
  if (name.ends_with("ffn_gate_inp.weight")) return 0.1f * projectionScale(tensor.dims[0]);
  if (name.find("_ctrl_") != std::string::npos) return projectionScale(tensor.dims[0]);
  if (name.ends_with("_conv.weight")) return 0.25f;
  // The queries' scale and the relative bias a tenth, so that a row's
  // attention spreads over its keys as a model's does rather than picking
  // one by a margin its rounding decides.
  if (name.ends_with("attn_q_clip.weight") || name.ends_with("attn_rel_bias.weight")) return 0.1f;
  return 1.0f;
}
// The weight layer of a tensor named blk.N.*.
inline uint32_t layerOf(const std::string &name) { return uint32_t(std::stoul(name.substr(4))); }

// The tensors of a GGUF of an inkling2_loop layout, as llama.cpp's converter
// writes Inkling-2-Mini's: Q8_0 projections, head and token table, BF16 sink
// and tap projections, NVFP4 experts beside a positive F32 .scale per expert
// but the last block's, Q5_K gate/up and Q8_0 down, F32 the rest; each of
// seeded random values (fixture's and floatValues', and llama.cpp's NVFP4
// rows of blockNvfp4Fixture), one seed each in order from seed + 1, in a
// checkpoint's sizes (projectionScale).
inline std::vector<Tensor> inkling2Tensors(const model::Inkling2LoopLayout &g, uint32_t seed) {
  std::vector<Tensor> tensors;
  const auto add = [&](const std::string &name, std::vector<uint64_t> dims, uint32_t type = model::ggml::kF32) {
    tensors.push_back({name, std::move(dims), type, {}});
  };
  const uint64_t hidden = g.hiddenSize, heads = g.attentionQueryHeads, kv = g.kvWidth(), taps = g.convolutionTaps;
  const uint64_t experts = g.experts, width = g.expertIntermediateSize;
  for (uint32_t layer = 0; layer < g.weightLayers; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    if (const uint32_t queries = g.mixerQueries(layer))
      add(p + "attn_res_query.weight", queries == 1 ? std::vector<uint64_t>{hidden} : std::vector<uint64_t>{hidden, 3});
    if (g.slides(layer)) {
      add(p + "attn_norm.weight", {hidden});
    } else {
      for (const char *input : {"q", "k", "v"}) add(p + "attn_" + input + "_inp_norm.weight", {hidden});
      add(p + "attn_k_conv_proj.weight", {hidden, g.convolutionTapsWidth()}, model::ggml::kBF16);
      add(p + "attn_v_conv_proj.weight", {hidden, g.convolutionTapsWidth()}, model::ggml::kBF16);
    }
    add(p + "attn_q.weight", {hidden, g.attentionWidth}, kQ8_0);
    add(p + "attn_gate.weight", {hidden, g.attentionWidth}, kQ8_0);
    add(p + "attn_rel.weight", {hidden, g.relativeWidth()}, kQ8_0);
    add(p + "attn_k.weight", {hidden, kv}, kQ8_0);
    add(p + "attn_v.weight", {hidden, kv}, kQ8_0);
    add(p + "attn_output.weight", {g.attentionWidth, hidden}, kQ8_0);
    add(p + "attn_sink_proj.weight", {hidden, heads}, model::ggml::kBF16);
    add(p + "attn_q_clip.weight", {g.attentionHeadDimension, heads});
    add(p + "attn_rel_clip.weight", {g.relativeRank, heads});
    add(p + "attn_rel_bias.weight", {g.relativeRank, g.relativeExtent});
    add(p + "attn_k_conv.weight", {taps, kv});
    add(p + "attn_v_conv.weight", {taps, kv});
    add(p + "attn_post_conv.weight", {taps, hidden});
    add(p + "ffn_post_conv.weight", {taps, hidden});
    add(p + "ffn_norm.weight", {hidden});
    add(p + "ffn_res_query.weight", {hidden});
    add(p + "ffn_gate_inp.weight", {hidden, experts});
    add(p + "exp_probs_b.bias", {experts});
    if (layer + g.blockResidualPeriod >= g.weightLayers) {
      add(p + "ffn_gate_up_exps.weight", {hidden, 2 * width, experts}, kQ5_K);
      add(p + "ffn_down_exps.weight", {width, hidden, experts}, kQ8_0);
    } else {
      add(p + "ffn_gate_up_exps.weight", {hidden, 2 * width, experts}, model::ggml::kNVFP4);
      add(p + "ffn_gate_up_exps.scale", {experts});
      add(p + "ffn_down_exps.weight", {width, hidden, experts}, model::ggml::kNVFP4);
      add(p + "ffn_down_exps.scale", {experts});
    }
    if (layer < g.loopFirstLayer || layer >= g.loopEndLayer) continue;
    for (const char *input : {"q", "k", "v", "ffn"}) {
      const std::string stem = p + "loop_" + input + "_ctrl_";
      add(stem + "down.weight", {2 * hidden, g.loopGateRank});
      add(stem + "forget.weight", {g.loopGateRank, hidden});
      add(stem + "gain.weight", {g.loopGateRank, hidden});
    }
    add(p + "loop_msg_query.weight", {hidden, 4});
  }
  add("output.weight", {hidden, g.vocabularySize}, kQ8_0);
  add("output_norm.weight", {hidden});
  add("output_res_query.weight", {hidden});
  add("token_embd.weight", {hidden, g.vocabularySize}, kQ8_0);
  add("token_embd_norm.weight", {hidden});
  // Each NVFP4 expert's multiplier (its .scale): a projection's size over the
  // root mean square of its random blocks' values.
  std::map<std::string, std::vector<float>> expertScales;
  for (Tensor &tensor : tensors) {
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    const uint64_t inputs = tensor.dims[0];
    std::mt19937 rng(++seed);
    if (tensor.type == model::ggml::kNVFP4) {
      tensor.data = blockNvfp4Fixture(uint32_t(rows), uint32_t(inputs), seed);
      const std::vector<float> values = gguf_reference::blockNvfp4Values(tensor.data);
      const uint64_t expertValues = values.size() / tensor.dims.back();
      std::vector<float> &scales = expertScales[tensor.name.substr(0, tensor.name.size() - 7)];
      for (uint64_t expert = 0; expert < tensor.dims.back(); ++expert) {
        double squares = 0.0;
        for (uint64_t i = 0; i < expertValues; ++i) {
          const double value = values[expert * expertValues + i];
          squares += value * value;
        }
        scales.push_back(projectionScale(inputs) * outputFactor(tensor.name) /
                         float(std::sqrt(squares / double(expertValues))));
      }
    } else if (tensor.type == model::ggml::kBF16) {
      tensor.data.resize(inputs * rows * 2);
      const std::vector<uint8_t> values = floatValues(inputs * rows, seed);
      for (uint64_t i = 0; i < inputs * rows; ++i)
        storeBf16(tensor.data, i, loadFloat(values, i) * projectionScale(inputs));
    } else if (tensor.type != model::ggml::kF32) {
      tensor.data = fixture(Fmt(gguf_format_of(tensor.type)), uint32_t(rows), uint32_t(inputs), seed);
      scaleBlocks(tensor, inputs, rng);
    } else {
      tensor.data = floatValues(inputs * rows, seed);
      if (tensor.name.ends_with(".scale")) {
        // Its expert tensor comes before it.
        const std::vector<float> &scales = expertScales.at(tensor.name.substr(0, tensor.name.size() - 6));
        for (uint64_t i = 0; i < inputs * rows; ++i) storeFloat(tensor.data, i, scales[i]);
      } else if (tensor.name.ends_with("exp_probs_b.bias")) {
        // The router's bias sets the selection apart by a tenth an expert, a
        // step each, past what its small scores move (inkling2FloatScale), so
        // that no row's top 8 is a near-tie the engine's rounding and the
        // reference's would settle apart: an output then follows its
        // mechanism alone, not a coin.
        for (uint64_t e = 0; e < inputs; ++e)
          storeFloat(tensor.data, e, 0.1f * float((e * 7 + layerOf(tensor.name)) % inputs) - 0.05f * float(inputs));
      } else {
        const float scale = inkling2FloatScale(tensor);
        for (uint64_t i = 0; i < inputs * rows; ++i) storeFloat(tensor.data, i, loadFloat(tensor.data, i) * scale);
      }
    }
  }
  return tensors;
}

// The header's keys of a GGUF of an inkling2_loop layout, as the converter
// writes them: the head counts one per weight layer, no RoPE, its sliding
// layers a bool each.
inline std::vector<test_gguf::Key> inkling2Metadata(const model::Inkling2LoopLayout &g) {
  const std::string arch(model::Inkling2LoopLayout::ggufArchitecture);
  std::vector<test_gguf::Key> keys{test_gguf::stringKey("general.architecture", arch)};
  const auto key = [&](const char *name, uint32_t value) {
    keys.push_back(test_gguf::uint32Key(arch + "." + name, value));
  };
  const auto real = [&](const char *name, float value) {
    keys.push_back(test_gguf::float32Key(arch + "." + name, value));
  };
  key("block_count", g.weightLayers);
  key("context_length", g.modelContextTokens);
  key("embedding_length", g.hiddenSize);
  key("vocab_size", g.vocabularySize);
  key("vocab_size_unpadded", g.vocabularyTokens);
  keys.push_back(test_gguf::int32ArrayKey(arch + ".attention.head_count",
                                          std::vector<int32_t>(g.weightLayers, int32_t(g.attentionQueryHeads))));
  keys.push_back(test_gguf::int32ArrayKey(arch + ".attention.head_count_kv",
                                          std::vector<int32_t>(g.weightLayers, int32_t(g.attentionKvHeads))));
  for (const char *name : {"attention.key_length", "attention.value_length", "attention.key_length_swa",
                           "attention.value_length_swa"})
    key(name, g.attentionHeadDimension);
  key("rope.dimension_count", 0);
  real("attention.layer_norm_rms_epsilon", 1e-6F);
  key("attention.sliding_window", g.slidingWindow);
  std::vector<bool> sliding;
  for (uint32_t layer = 0; layer < g.weightLayers; ++layer) sliding.push_back(g.slides(layer));
  keys.push_back(test_gguf::boolArrayKey(arch + ".attention.sliding_window_pattern", sliding));
  key("attention.relative_length", g.relativeRank);
  key("attention.relative_extent", g.relativeExtent);
  real("attention.relative_clip", g.relativeClip);
  real("attention.qk_clip", g.queryClip);
  key("attention.log_scale_floor", g.logScalingFloor);
  real("attention.log_scale_alpha", g.logScalingAlpha);
  key("shortconv.l_cache", g.convolutionTaps);
  key("attn_res.block_size", g.blockResidualPeriod);
  key("loop.start", g.loopFirstLayer);
  key("loop.end", g.loopEndLayer);
  key("loop.count", g.loopPasses);
  key("loop.gate_rank", g.loopGateRank);
  keys.push_back(test_gguf::boolKey(arch + ".loop.varying", false));
  real("activation.situ_beta", g.expertGateSoftcap);
  real("activation.situ_linear_beta", g.expertUpSoftcap);
  real("final_logit_softcapping", g.logitSoftcap);
  key("expert_count", g.experts);
  key("expert_used_count", g.expertsPerToken);
  key("expert_feed_forward_length", g.expertIntermediateSize);
  key("feed_forward_length", g.expertIntermediateSize);
  real("expert_weights_scale", g.expertRoutedScale);
  return keys;
}

// A small Inkling-2-Mini target in the engine's shapes, which the engine
// test runs and its reference reads, of every kind of layer the model runs:
// 24 query heads over 8 KV heads of 128, a window of 1024 keys and relative
// extent 1024, rank 16, 4 taps, gate rank 64, its log scaling's floor; 12
// weight layers of 256 in blocks of 3, layers 3-8 looped three times (24
// applications: 4 stack planes and 2 of the pass before, as the model's), the
// last layer of each block attending every key (2, 5, 8, 11); 16 experts of
// 320, top 8, so that the gate and up of 256 inputs and the down of 320 take
// the model's two NVFP4 forms; 320 vocabulary rows, 314 of them tokens.
inline model::Inkling2LoopLayout inkling2EngineLayout() {
  model::Inkling2LoopLayout g;
  g.weightLayers = 12;
  g.loopFirstLayer = 3;
  g.loopEndLayer = 9;
  g.loopPasses = 3;
  g.blockResidualPeriod = 3;
  g.slidingWeightLayers = 0x6DB;
  g.layers = g.schedule().applications();
  g.hiddenSize = 256;
  g.vocabularySize = 320;
  g.vocabularyTokens = 314;
  g.slidingLayers = g.schedule().slidingApplications();
  g.unrotatedLayers = (uint64_t{1} << g.layers) - 1;
  g.experts = 16;
  g.expertsPerToken = 8;
  g.expertIntermediateSize = 320;
  return g;
}

// Writes a GGUF of an inkling2_loop layout of seeded values (inkling2Tensors).
inline void writeInkling2Gguf(const std::filesystem::path &path, const model::Inkling2LoopLayout &g, uint32_t seed) {
  splash::test::writeFile(path, test_gguf::file(inkling2Metadata(g), inkling2Tensors(g, seed)));
}

} // namespace gguf_fixtures
