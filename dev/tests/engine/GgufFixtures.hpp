#pragma once

// What the GGUF reference, planner and preparation tests share: their
// reporting, the golden hashes (dev/tests/fixtures/weight-goldens), random
// native blocks and F32 values, and small qwen35 and qwen35moe GGUF targets
// in the shapes llama.cpp writes.

#import <Foundation/Foundation.h>

#include "GgufFormatReference.hpp"
#include "TestFiles.hpp"
#include "TestGguf.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/WeightSource.hpp"
#include "model/WeightStore.hpp"
#include "model/StateLayout.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gguf_fixtures {

namespace model = splash::model;
namespace test_gguf = splash::test::gguf;
using gguf_reference::Fmt;
using test_gguf::Tensor;

// The id of the ggml type named `name` in model::kGgmlTypes; an unknown name
// does not compile.
consteval uint32_t ggmlType(std::string_view name) {
  for (const auto &[id, traits] : model::kGgmlTypes)
    if (traits.name == name) return id;
  throw std::invalid_argument("unknown ggml type");
}
inline constexpr uint32_t kQ8_0 = ggmlType("Q8_0"), kQ3_K = ggmlType("Q3_K"), kQ4_K = ggmlType("Q4_K"),
                          kQ5_K = ggmlType("Q5_K"), kQ6_K = ggmlType("Q6_K"), kIQ4_NL = ggmlType("IQ4_NL"),
                          kIQ3_S = ggmlType("IQ3_S"), kIQ4_XS = ggmlType("IQ4_XS");

inline int failures = 0;

// One tensor of the MLX quantization fixture (dev/tests/fixtures/mlx-quantization), which
// dev/tools/mlx_quantization_fixture.py writes with MLX: its mode (affine, mxfp4 or nvfp4), bits, group size and
// shape, MLX's packed codes, scales (bf16, or mxfp4's E8M0 and nvfp4's E4M3 bytes) and an affine tensor's bf16
// biases, and MLX's reading of it, an affine tensor's codes and the fp32 values.
struct MlxTensor {
  std::string mode;
  uint32_t bits = 0, group = 0, rows = 0, columns = 0;
  std::vector<uint8_t> weight, scales, biases, codes, values;
  [[nodiscard]] bool affine() const { return mode == "affine"; }
  // Its format (metal/abi/QuantFormat.h).
  [[nodiscard]] Fmt format() const {
    return affine() ? Fmt(quant_affine_format_of(bits, group))
                    : mode == "nvfp4" ? gguf_reference::NVFP4 : gguf_reference::MXFP4;
  }
  [[nodiscard]] std::vector<uint8_t> native() const {
    return gguf_reference::mlxNative(format(), rows, columns, weight, scales, biases);
  }
};
inline std::vector<MlxTensor> mlxFixture(const char *path) {
  NSData *data = [NSData dataWithContentsOfFile:@(path)];
  NSDictionary *fixture = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
  if (![fixture isKindOfClass:NSDictionary.class]) throw std::runtime_error(std::string("unreadable ") + path);
  const auto bytes = [](NSString *hex) {
    std::vector<uint8_t> result(hex.length / 2);
    const char *text = hex.UTF8String;
    for (size_t i = 0; i < result.size(); ++i)
      result[i] = uint8_t(std::stoul(std::string(text + 2 * i, 2), nullptr, 16));
    return result;
  };
  std::vector<MlxTensor> tensors;
  for (NSDictionary *entry in fixture[@"formats"]) {
    MlxTensor &t = tensors.emplace_back();
    t.mode = [entry[@"mode"] UTF8String];
    t.bits = [entry[@"bits"] unsignedIntValue];
    t.group = [entry[@"group"] unsignedIntValue];
    t.rows = [entry[@"rows"] unsignedIntValue];
    t.columns = [entry[@"columns"] unsignedIntValue];
    t.weight = bytes(entry[@"weight"]);
    t.scales = bytes(entry[@"scales"]);
    t.values = bytes(entry[@"values"]);
    if (t.affine()) {
      t.biases = bytes(entry[@"biases"]);
      t.codes = bytes(entry[@"codes"]);
    }
  }
  return tensors;
}

inline void check(bool ok, const std::string &what) {
  std::printf("%-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  failures += !ok;
}

// Runs one group of checks; an exception fails the group, not the rest.
template <class F> void guarded(const std::string &what, F run) {
  try {
    run();
  } catch (const std::exception &error) {
    check(false, what + ": " + error.what());
  }
}

// The hashes of one section of the goldens file, nested keys joined by '/'.
using Goldens = std::map<std::string, std::string>;
inline Goldens goldens(const char *path, NSString *section) {
  NSData *data = [NSData dataWithContentsOfFile:@(path)];
  NSDictionary *all = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
  if (![all isKindOfClass:NSDictionary.class] || ![all[section] isKindOfClass:NSDictionary.class])
    throw std::runtime_error(std::string("no ") + section.UTF8String + " in " + path);
  Goldens hashes;
  const auto flatten = [&](auto &self, NSDictionary *object, const std::string &prefix) -> void {
    for (NSString *key in object) {
      const std::string name = prefix + key.UTF8String;
      if ([object[key] isKindOfClass:NSDictionary.class]) self(self, object[key], name + "/");
      else hashes[name] = [object[key] UTF8String];
    }
  };
  flatten(flatten, all[section], "");
  return hashes;
}

// Checks bytes against the golden `name` and prints their hash when it differs.
inline void checkGolden(const Goldens &hashes, const std::string &name, std::span<const uint8_t> bytes,
                        const std::string &what) {
  const std::string actual = model::weightDigest(bytes);
  const auto golden = hashes.find(name);
  const bool matches = golden != hashes.end() && golden->second == actual;
  check(matches, what);
  if (!matches) std::printf("  %s is now %s\n", name.c_str(), actual.c_str());
}

// Every byte random and every half scale a random finite half: either sign,
// zero and subnormal included; an MLX affine format's bf16 scales and biases
// random finite bf16s; NVFP4's and FP8's blocks as makeNative draws them.
inline std::vector<uint8_t> fixture(Fmt f, uint32_t rows, uint32_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  if (f == gguf_reference::NVFP4 || f == gguf_reference::FP8) return gguf_reference::makeNative(f, rows, K, rng);
  if (gguf_reference::affine(f)) {
    std::vector<uint8_t> native((size_t)rows * gguf_reference::rowBytes(f, K));
    for (auto &b : native) b = (uint8_t)rng();
    for (size_t at = 0; at < native.size(); at += kQuantFormats[f].block_bytes)
      for (size_t field = 0; field < 4; field += 2)
        if ((native[at + field + 1] & 0x7F) == 0x7F && (native[at + field] & 0x80)) native[at + field + 1] ^= 1;
    return native;
  }
  return gguf_reference::makeNative(f, rows, K, rng, [&] {
    uint16_t h;
    do h = static_cast<uint16_t>(rng());
    while ((h & 0x7C00) == 0x7C00);
    return h;
  });
}

// The GGUF tensor type of a format's tensors: its table type, or llama.cpp's
// NVFP4 for NVFP4, whose native rows the loader builds from it; none for the
// loader's other formats (MLX affine, FP8).
inline std::optional<uint32_t> ggufType(uint32_t format) {
  if (format == GGUF_FMT_NVFP4) return model::ggml::kNVFP4;
  if (quant_loader_format(format)) return std::nullopt;
  return kQuantFormats[format].ggml_type;
}

// block_nvfp4 rows of llama.cpp's NVFP4 (ggml type 40), every byte random,
// so their scales take every byte: 0x7F, which UE4M3 reads as 0, and those
// of bit 7, which it ignores, included.
inline std::vector<uint8_t> blockNvfp4Fixture(uint32_t rows, uint32_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> blocks(size_t(rows) * K / 64 * gguf_reference::kBlockNvfp4Bytes);
  for (auto &b : blocks) b = uint8_t(rng());
  return blocks;
}

// F32 values in [-1, -0.5] and [0.5, 1], optionally exact bf16 values (the
// low half of every word zero), as a bf16 checkpoint converted to F32 has.
inline std::vector<uint8_t> floatValues(uint64_t count, uint32_t seed, bool bfloat16 = false) {
  std::vector<uint8_t> bytes(count * 4);
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t bits = 0x3F000000u | uint32_t((i * 2654435761u + seed * 40503u) & 0x7FFFFFu);
    if ((i + seed) % 3 == 0) bits |= 0x80000000u;
    if (bfloat16) bits &= 0xFFFF0000u;
    std::memcpy(bytes.data() + i * 4, &bits, 4);
  }
  return bytes;
}

// The ggml type of each quantized tensor, by name or by its name without the
// "blk.N." prefix; every other tensor is F32.
using TensorTypes = std::map<std::string, uint32_t>;

inline uint32_t typeOf(const TensorTypes &types, const std::string &name) {
  if (const auto found = types.find(name); found != types.end()) return found->second;
  const size_t role = name.starts_with("blk.") ? name.find('.', 4) + 1 : 0;
  const auto found = types.find(name.substr(role));
  return found == types.end() ? model::ggml::kF32 : found->second;
}

// Every tensor of the geometry, zero, in the order llama.cpp writes them.
inline std::vector<Tensor> targetTensors(const model::QwenTargetDimensions &g, const TensorTypes &types) {
  std::vector<Tensor> tensors;
  const auto add = [&](std::string name, std::vector<uint64_t> dims) {
    const uint32_t type = typeOf(types, name);
    const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(type);
    uint64_t elements = 1;
    for (uint64_t dim : dims) elements *= dim;
    tensors.push_back(
        {name, std::move(dims), type, test_gguf::Bytes(elements / traits.blockElements * traits.blockBytes)});
  };
  const uint64_t hidden = g.hiddenSize, valueRows = uint64_t{g.gdnValueHeads} * g.gdnHeadDimension;
  const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
  for (uint32_t layer = 0; layer < g.layers; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    add(p + "attn_norm.weight", {hidden});
    if (g.isFullAttentionLayer(layer)) {
      add(p + "attn_q.weight", {hidden, 2ull * g.attentionWidth}); // queries and their gate
      add(p + "attn_k.weight", {hidden, kvRows});
      add(p + "attn_v.weight", {hidden, kvRows});
      add(p + "attn_q_norm.weight", {g.attentionHeadDimension});
      add(p + "attn_k_norm.weight", {g.attentionHeadDimension});
      add(p + "attn_output.weight", {g.attentionWidth, hidden});
    } else {
      add(p + "attn_qkv.weight", {hidden, g.convolutionDimension});
      add(p + "attn_gate.weight", {hidden, valueRows});
      add(p + "ssm_beta.weight", {hidden, g.gdnValueHeads});
      add(p + "ssm_alpha.weight", {hidden, g.gdnValueHeads});
      add(p + "ssm_conv1d.weight", {model::kGdnConvolutionTaps, g.convolutionDimension});
      add(p + "ssm_a", {g.gdnValueHeads});
      add(p + "ssm_dt.bias", {g.gdnValueHeads});
      add(p + "ssm_norm.weight", {g.gdnHeadDimension});
      add(p + "ssm_out.weight", {valueRows, hidden});
    }
    add(p + "post_attention_norm.weight", {hidden});
    if (g.ffnKind == model::QwenFfnKind::SparseMoe) {
      const uint64_t experts = g.experts, width = g.expertIntermediateSize;
      add(p + "ffn_gate_inp.weight", {hidden, experts});
      add(p + "ffn_gate_exps.weight", {hidden, width, experts});
      add(p + "ffn_up_exps.weight", {hidden, width, experts});
      add(p + "ffn_down_exps.weight", {width, hidden, experts});
      add(p + "ffn_gate_shexp.weight", {hidden, width});
      add(p + "ffn_up_shexp.weight", {hidden, width});
      add(p + "ffn_down_shexp.weight", {width, hidden});
      add(p + "ffn_gate_inp_shexp.weight", {hidden});
    } else {
      add(p + "ffn_gate.weight", {hidden, g.intermediateSize});
      add(p + "ffn_up.weight", {hidden, g.intermediateSize});
      add(p + "ffn_down.weight", {g.intermediateSize, hidden});
    }
  }
  add("output_norm.weight", {hidden});
  add("output.weight", {hidden, g.vocabularySize});
  add("token_embd.weight", {hidden, g.vocabularySize});
  return tensors;
}

inline bool isNorm(std::string_view name) { return name.ends_with("norm.weight"); }

// The F32 tensors the kernels read as bf16, which must hold bf16 values.
inline bool readAsBfloat16(const Tensor &tensor) {
  return tensor.name.ends_with("ssm_conv1d.weight") || tensor.name.ends_with("ssm_dt.bias");
}

inline Tensor &tensorNamed(std::vector<Tensor> &tensors, const std::string &name) {
  const auto found = std::find_if(tensors.begin(), tensors.end(), [&](const Tensor &t) { return t.name == name; });
  if (found == tensors.end()) throw std::runtime_error("no fixture tensor " + name);
  return *found;
}

// A small target: its geometry and its tensors.
struct SmallTarget {
  model::QwenTargetDimensions geometry;
  std::vector<Tensor> tensors;
  [[nodiscard]] test_gguf::Bytes &data(const std::string &name) { return tensorNamed(tensors, name).data; }
};

// Random values for every tensor, one seed each in file order from seed + 1:
// quantized tensors in their format's blocks, the others F32, bf16-exact
// where the kernels read them as bf16.
inline void randomize(std::vector<Tensor> &tensors, uint32_t seed) {
  for (Tensor &tensor : tensors) {
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    const uint32_t format = gguf_format_of(tensor.type);
    tensor.data = format == GGUF_FMT_COUNT ? floatValues(tensor.dims[0] * rows, ++seed, readAsBfloat16(tensor))
                                           : fixture(Fmt(format), uint32_t(rows), uint32_t(tensor.dims[0]), ++seed);
  }
}

// The targets the golden images were recorded from, seeds included. dense: a
// GDN layer and a full-attention layer in eight formats (Q4_K, Q5_K, Q6_K,
// Q3_K, Q8_0, IQ4_XS, IQ4_NL and IQ3_S; gguf-preparation's repack check covers
// every format), Q8_0 alpha/beta and permuted value-head rows, seeds 901-928.
// moe: one qwen35moe layer with F32 alpha/beta, router and shared-expert
// scalar gate and 3-D expert tensors, its seeds after dense's and output_norm
// first in its file.
inline SmallTarget smallTarget(bool moe) {
  SmallTarget target;
  model::QwenTargetDimensions &g = target.geometry;
  g.hiddenSize = 512;
  g.vocabularySize = 256;
  g.gdnKeyHeads = 4;
  g.gdnHeadDimension = 64;
  g.attentionWidth = 512; // two query heads of 256
  g.attentionKvHeads = 2;
  g.attentionHeadDimension = 256;
  g.rotaryPairs = 32; // 64 of the 256 dimensions, at the real targets' base
  g.rotaryTheta = 1e7F;
  g.fullAttentionPeriod = 2; // layer 1
  if (moe) {
    g.layers = 1;
    g.gdnValueHeads = 8;
    g.convolutionDimension = 1024; // q and k of 4 heads, v of 8
    g.ffnKind = model::QwenFfnKind::SparseMoe;
    g.experts = 4;
    g.expertsPerToken = 2;
    g.expertIntermediateSize = 256;
    target.tensors = targetTensors(g, {{"attn_qkv.weight", kQ8_0},
                                       {"attn_gate.weight", kQ6_K},
                                       {"ssm_out.weight", kQ4_K},
                                       {"ffn_gate_exps.weight", kQ4_K},
                                       {"ffn_up_exps.weight", kQ4_K},
                                       {"ffn_down_exps.weight", kQ5_K},
                                       {"ffn_gate_shexp.weight", kQ8_0},
                                       {"ffn_up_shexp.weight", kQ8_0},
                                       {"ffn_down_shexp.weight", kQ8_0},
                                       {"output.weight", kQ6_K},
                                       {"token_embd.weight", kQ8_0}});
    std::rotate(target.tensors.begin(), target.tensors.end() - 3, target.tensors.end() - 2);
    randomize(target.tensors, 928);
  } else {
    g.layers = 2;
    g.gdnValueHeads = 12;
    g.convolutionDimension = 1280; // q and k of 4 heads, v of 12
    g.intermediateSize = 256;
    target.tensors = targetTensors(g, {{"blk.0.attn_qkv.weight", kQ4_K},
                                       {"blk.0.attn_gate.weight", kQ5_K},
                                       {"blk.0.ssm_beta.weight", kQ8_0},
                                       {"blk.0.ssm_alpha.weight", kQ8_0},
                                       {"blk.0.ssm_out.weight", kIQ4_XS},
                                       {"blk.0.ffn_gate.weight", kQ3_K},
                                       {"blk.0.ffn_up.weight", kIQ3_S},
                                       {"blk.0.ffn_down.weight", kIQ4_NL},
                                       {"blk.1.attn_q.weight", kQ6_K},
                                       {"blk.1.attn_k.weight", kQ8_0},
                                       {"blk.1.attn_v.weight", kQ4_K},
                                       {"blk.1.attn_output.weight", kQ5_K},
                                       {"blk.1.ffn_gate.weight", kQ4_K},
                                       {"blk.1.ffn_up.weight", kQ6_K},
                                       {"blk.1.ffn_down.weight", kQ8_0},
                                       {"output.weight", kQ6_K},
                                       {"token_embd.weight", kQ6_K}});
    randomize(target.tensors, 900);
  }
  return target;
}

// The architecture metadata a GGUF of the geometry declares.
inline std::vector<test_gguf::Key> metadata(const model::QwenTargetDimensions &g) {
  const std::string arch = model::gguf::architecture(g.ffnKind);
  std::vector<test_gguf::Key> keys{test_gguf::stringKey("general.architecture", arch)};
  const auto key = [&](const char *name, uint32_t value) {
    keys.push_back(test_gguf::uint32Key(arch + "." + name, value));
  };
  key("block_count", g.layers);
  key("embedding_length", g.hiddenSize);
  key("attention.head_count", g.attentionWidth / g.attentionHeadDimension);
  key("attention.head_count_kv", g.attentionKvHeads);
  key("attention.key_length", g.attentionHeadDimension);
  key("attention.value_length", g.attentionHeadDimension);
  key("rope.dimension_count", 2 * g.rotaryPairs);
  keys.push_back(test_gguf::float32Key(arch + ".rope.freq_base", g.rotaryTheta));
  keys.push_back(test_gguf::float32Key(arch + ".attention.layer_norm_rms_epsilon", 1e-6F));
  key("full_attention_interval", g.fullAttentionPeriod);
  key("ssm.conv_kernel", model::kGdnConvolutionTaps);
  key("ssm.group_count", g.gdnKeyHeads);
  key("ssm.time_step_rank", g.gdnValueHeads);
  key("ssm.state_size", g.gdnHeadDimension);
  key("ssm.inner_size", g.gdnValueHeads * g.gdnHeadDimension);
  if (g.ffnKind == model::QwenFfnKind::SparseMoe) {
    key("expert_count", g.experts);
    key("expert_used_count", g.expertsPerToken);
    key("expert_feed_forward_length", g.expertIntermediateSize);
    key("expert_shared_feed_forward_length", g.expertIntermediateSize);
  } else {
    key("feed_forward_length", g.intermediateSize);
  }
  return keys;
}

// Writes a GGUF of the tensors declaring the geometry.
inline void writeGguf(const std::filesystem::path &path, const std::vector<Tensor> &tensors,
                      const model::QwenTargetDimensions &geometry) {
  splash::test::writeFile(path, test_gguf::file(metadata(geometry), tensors));
}

// The plan's copy of the named tensor, or nullptr.
inline const model::gguf::Copy *copyOf(const model::gguf::Image &image, const std::string &name) {
  const auto found = std::find_if(image.copies.begin(), image.copies.end(),
                                  [&](const model::gguf::Copy &copy) { return copy.source.name == name; });
  return found == image.copies.end() ? nullptr : &*found;
}

// The plan's repack whose first source is the named tensor, or nullptr.
inline const model::gguf::Repack *repackOf(const model::gguf::Image &image, const std::string &name) {
  const auto found = std::find_if(image.repacks.begin(), image.repacks.end(), [&](const model::gguf::Repack &repack) {
    return !repack.sources.empty() && repack.sources.front().name == name;
  });
  return found == image.repacks.end() ? nullptr : &*found;
}

} // namespace gguf_fixtures
