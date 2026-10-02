// The rules an upstream model's configuration is held to before any of its
// weights load (model/ModelDescriptor.mm), at every start and by the
// installer's check before any weight download, on configs built like each
// supported family's: the MLX target's config.json and its DFlash2 draft's,
// read from FIXTURES (dev/tests/fixtures/model-configs/<family>), and the
// config the installer derives from a GGUF's metadata with the metadata it
// copies (gguf-metadata.json there).
//
//   model-configuration FIXTURES

#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/WeightStore.hpp"

#include <array>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace {

using namespace splash;
using test::rejects;
using test::require;

// An upstream model as its assembly records it: model.json, the target's
// config.json and the draft's.
struct SourceModel final {
  std::string record, config, draft;

  // This model with the first `from` of one of its files replaced by `to`.
  [[nodiscard]] SourceModel with(std::string SourceModel::*file, std::string_view from,
                                 std::string_view to) const {
    SourceModel result = *this;
    std::string &text = result.*file;
    const size_t at = text.find(from);
    require(at != std::string::npos, "the source model has no " + std::string(from));
    text.replace(at, from.size(), to);
    return result;
  }
};

std::string text(const std::filesystem::path &path) {
  const auto bytes = test::readFile(path);
  return {bytes.begin(), bytes.end()};
}

// The family's MLX model as the fixtures hold it, text only.
SourceModel mlxModel(const std::filesystem::path &fixtures, std::string_view family) {
  return {R"({"version":1,"model":"community/fine-tune","target_format":"mlx-affine","vision_format":"none"})",
          text(fixtures / family / "config.json"), text(fixtures / family / "draft" / "config.json")};
}

// The config the installer derives from a GGUF of each family's target.
constexpr std::string_view kDenseGgufConfig =
    R"({"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":5120,)"
    R"("max_position_embeddings":262144,"num_attention_heads":24,"num_key_value_heads":4,)"
    R"("head_dim":256,"num_hidden_layers":64,"vocab_size":248320}})";
constexpr std::string_view kMoeGgufConfig =
    R"({"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":2048,)"
    R"("max_position_embeddings":262144,"num_attention_heads":16,"num_key_value_heads":2,)"
    R"("head_dim":256,"num_hidden_layers":40,"vocab_size":248320}})";

// The descriptor inspection makes of model's assembly.
model::ModelDescriptor inspect(const SourceModel &model) {
  const test::TemporaryDirectory root("splash-source-model");
  std::filesystem::create_directory(root.path() / "draft");
  test::writeFile(root.path() / "model.json", model.record);
  test::writeFile(root.path() / "config.json", model.config);
  test::writeFile(root.path() / "draft" / "config.json", model.draft);
  return model::inspectModelRoot(root.path());
}

void refuses(const SourceModel &model, std::string_view error, const std::string &message) {
  rejects([&] { static_cast<void>(inspect(model)); }, error, message);
}

// Each family's MLX config and draft config make its descriptor, with the
// vision tower when the record names one.
void testFamilies(const std::filesystem::path &fixtures) {
  const SourceModel dense = mlxModel(fixtures, "qwen3.8-27b");
  const model::ModelDescriptor denseDescriptor = inspect(dense);
  require(denseDescriptor.name == "community/fine-tune" &&
              std::holds_alternative<model::Qwen3_8Layout>(denseDescriptor.target) &&
              denseDescriptor.targetSource == model::TargetSource::Safetensors && !denseDescriptor.hasVision() &&
              denseDescriptor.draft == model::kQwen3_8DraftLayout,
          "an MLX Qwen3.8-27B made another descriptor");
  // The sources' identity is the digest of the record naming them.
  require(denseDescriptor.sourceIdentity == model::weightDigest(dense.record) &&
              inspect(dense.with(&SourceModel::record, "community/fine-tune", "community/other"))
                      .sourceIdentity != denseDescriptor.sourceIdentity,
          "the sources' identity is not the digest of the record naming them");
  require(inspect(dense.with(&SourceModel::record, R"("none")", R"("safetensors")")).visionSource ==
              model::VisionSource::Safetensors,
          "an MLX Qwen3.8-27B with its vision tower made another descriptor");
  const SourceModel moe = mlxModel(fixtures, "qwen3.6-35b-a3b");
  const model::ModelDescriptor moeDescriptor =
      inspect(moe.with(&SourceModel::record, R"("none")", R"("safetensors")"));
  require(std::holds_alternative<model::Qwen3_6MoeLayout>(moeDescriptor.target) &&
              moeDescriptor.visionSource == model::VisionSource::Safetensors &&
              moeDescriptor.draft == model::kQwen3_6MoeDraftLayout,
          "an MLX Qwen3.6-35B-A3B made another descriptor");
  // The model type names the family; the sizes every source's config states
  // tell its target from other models of the architecture, which Splash does
  // not serve, such as Qwen3.5-4B.
  constexpr std::string_view supported =
      "; supported: Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next";
  struct Refused final {
    std::string_view from, to, difference;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {R"("hidden_size": 5120)", R"("hidden_size": 2560)", "text config hidden_size: MLX 2560, Qwen3.8-27B 5120"},
           {R"("num_hidden_layers": 64)", R"("num_hidden_layers": 32)",
            "text config num_hidden_layers: MLX 32, Qwen3.8-27B 64"},
           {R"("model_type": "qwen3_5_text")", R"("model_type": "gemma3_text")", "text model type gemma3_text"},
           {R"("text_config")", R"("unused")", "its config has no text_config"},
       })
    refuses(dense.with(&SourceModel::config, refused.from, refused.to),
            "no supported model has this architecture (" + std::string(refused.difference) + ")" +
                std::string(supported),
            "a config of another model was accepted with " + std::string(refused.to));
}

// A descriptor states where its weights come from: one that leaves either
// source unset, which hasVision would read as a vision tower, is not valid.
void testSources() {
  const auto described = [](model::TargetSource target, model::VisionSource vision) {
    return model::makeModelDescriptor("sources", model::Qwen3_8Layout{}, model::kQwen3_8DraftLayout,
                                      model::kQwen3_8VisionLayout, target, vision);
  };
  require(described(model::TargetSource::Safetensors, model::VisionSource::None).valid(),
          "a descriptor of its sources was not valid");
  require(!described(model::TargetSource{}, model::VisionSource::None).valid() &&
              !described(model::TargetSource::Safetensors, model::VisionSource{}).valid(),
          "a descriptor without a source was valid");
}

// Each config is checked where the descriptor is made, once, and each number
// by one rule: a JSON number, never a boolean, of the model's value, which a
// whole number may also spell as a float.
void testOneRulePerValue(const std::filesystem::path &fixtures) {
  const SourceModel source = mlxModel(fixtures, "qwen3.8-27b");
  static_cast<void>(
      inspect(source.with(&SourceModel::config, R"("rope_theta": 10000000)", R"("rope_theta": 1e7)")
                  .with(&SourceModel::draft, R"("block_size": 8)", R"("block_size": 8.0)")));
  // Transformers also reads the rope type from the older `type` key.
  static_cast<void>(
      inspect(source.with(&SourceModel::config, R"("rope_type": "default")", R"("type": "default")")));
  // A GGUF's config, which the installer derives from its metadata, holds
  // only the sizes the descriptor shares with it.
  SourceModel gguf = source.with(&SourceModel::record, R"("mlx-affine")", R"("gguf")");
  gguf.config = kDenseGgufConfig;
  require(inspect(gguf).targetSource == model::TargetSource::Gguf, "a GGUF's derived config was refused");
  // A difference names the source that states the value beside the family.
  refuses(gguf.with(&SourceModel::config, R"("hidden_size":5120)", R"("hidden_size":4096)"),
          "(text config hidden_size: GGUF 4096, Qwen3.8-27B 5120)", "a GGUF of another width was accepted");

  struct Refused final {
    std::string SourceModel::*file;
    std::string_view from, to, error;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {&SourceModel::record, R"("version":1)", R"("version":2)",
            "model record version mismatch: assembly 2, runtime 1"},
           {&SourceModel::config, R"("num_hidden_layers": 64)", R"("num_hidden_layers": true)",
            "text config num_hidden_layers must be a number"},
           {&SourceModel::config, R"("head_dim": 256,)", "", "text config head_dim must be a number"},
           {&SourceModel::config, R"("linear_num_value_heads": 48)", R"("linear_num_value_heads": 47)",
            "text config linear_num_value_heads mismatch: MLX 47, runtime 48"},
           {&SourceModel::config, R"("attn_output_gate": true)", R"("attn_output_gate": 1)",
            "text config attn_output_gate must be true"},
           {&SourceModel::config, R"("rope_type": "default")", R"("type": "yarn")",
            "text config rope_parameters type mismatch"},
           {&SourceModel::config, R"("full_attention")", R"("linear_attention")",
            "text config layer_types 3 must be full_attention"},
           {&SourceModel::draft, "[\n      5,", "[\n      5.9,",
            "draft config target_layer_ids 0 mismatch: checkpoint 5.9, runtime 5"},
           {&SourceModel::draft, "61\n", "61,\n      1\n",
            "draft config target_layer_ids count mismatch: checkpoint 6, runtime 5"},
           {&SourceModel::draft, R"("rope_theta": 10000000)", R"("rope_theta": 1000000)",
            "draft config rope_parameters rope_theta mismatch"},
           {&SourceModel::draft, R"("selector_top_k": 16)", R"("selector_top_k": 8)",
            "draft config dflash_config selector_top_k mismatch"},
           {&SourceModel::draft, R"("is_causal": false)", R"("is_causal": 0)", "draft config is_causal must be false"},
       })
    refuses(source.with(refused.file, refused.from, refused.to), refused.error,
            "an upstream config was accepted with " + std::string(refused.to));
  // A file past a megabyte is not a record or a config, and is not read.
  refuses(source.with(&SourceModel::config, R"("architectures")",
                      R"("padding": ")" + std::string(1 << 20, ' ') + R"(", "architectures")"),
          "upstream model config exceeds 1048576 bytes", "an oversized config was read");
}

// The model with a transformers quantization_config in place of MLX's
// objects: its "quantization" object renamed, which no check reads, and its
// "quantization_config" object replaced.
SourceModel withQuantizationConfig(const SourceModel &model, std::string_view quantizationConfig) {
  SourceModel result = model.with(&SourceModel::config, R"("quantization": {)", R"("unused": {)");
  std::string &config = result.config;
  const size_t at = config.find(R"("quantization_config": {)");
  require(at != std::string::npos, "the source model has no quantization_config");
  size_t end = config.find('{', at);
  for (int depth = 0; end < config.size(); ++end)
    if (config[end] == '{')
      ++depth;
    else if (config[end] == '}' && --depth == 0)
      break;
  config.replace(at, end + 1 - at, R"("quantization_config": )" + std::string(quantizationConfig));
  return result;
}

// An MLX target's quantization is MLX's own object: the object and each
// module's own entry name affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64
// or 128, mxfp4 or nvfp4; an entry is affine unless it names its mode, and
// takes its mode's default bits and group size (affine 4 and 64, mxfp4 4 and
// 32, nvfp4 4 and 16) when it omits them, as MLX loads it. The entry false,
// which MLX writes for a module it leaves unquantized, is refused for a
// projection and the head, and accepted for the router, the shared-expert
// gate, GDN alpha and beta and the token table. A Model Optimizer target's is
// its quantization_config: NVFP4 in groups of 16 or per-tensor FP8 for each
// layer it quantizes, every projection and the head among them. So is a
// compressed-tensors target's: each config group NVFP4 or FP8 per channel or
// tensor, every projection and the head a group's target that no ignore entry
// names. A GGUF target has none (above).
void testQuantization(const std::filesystem::path &fixtures) {
  const SourceModel dense = mlxModel(fixtures, "qwen3.8-27b");
  const SourceModel moe = mlxModel(fixtures, "qwen3.6-35b-a3b");
  static_cast<void>(inspect(dense));
  static_cast<void>(inspect(moe));
  // The dense object and the MoE's leading fields, as mxfp4.
  const std::string_view affine = R"("group_size": 64,
    "bits": 4,
    "mode": "affine")";
  const std::string_view mxfp4 = R"("group_size": 32,
    "bits": 4,
    "mode": "mxfp4")";
  const std::string_view nvfp4 = R"("group_size": 16,
    "bits": 4,
    "mode": "nvfp4")";
  // The MoE router's entry.
  const std::string_view router = R"("group_size": 64,
      "bits": 8)";
  // The object's first key, before which a module's entry is added.
  const std::string_view object = R"("quantization": {)";
  // The MoE's first router and shared-expert gate entries.
  const std::string_view routerEntry = R"("language_model.model.layers.0.mlp.gate": {
      "group_size": 64,
      "bits": 8
    })";
  const std::string_view sharedGateEntry = R"("language_model.model.layers.0.mlp.shared_expert_gate": {
      "group_size": 64,
      "bits": 8
    })";
  const std::string unquantizedTable = std::string(object) + R"("language_model.model.embed_tokens": false,)";
  struct Accepted final {
    const SourceModel &model;
    std::string_view from, to;
  };
  for (const Accepted &accepted : std::initializer_list<Accepted>{
           {dense, R"("bits": 4)", R"("bits": 8)"},
           {dense, R"("bits": 4)", R"("bits": 3)"},
           {dense, R"("group_size": 64)", R"("group_size": 128)"},
           {dense, affine, mxfp4},
           {dense, affine, nvfp4},
           // The router 4-bit; under an mxfp4 object, its 8-bit entry stays affine.
           {moe, R"("bits": 8)", R"("bits": 4)"},
           {moe, affine, mxfp4},
           // An entry of only its bits takes affine's group of 64.
           {moe, router, R"("bits": 8)"},
           // An entry of only its mode: mxfp4, 4-bit in groups of 32.
           {moe, router, R"("mode": "mxfp4")"},
           // Modules the images also read unquantized (as F32).
           {moe, routerEntry, R"("language_model.model.layers.0.mlp.gate": false)"},
           {moe, sharedGateEntry, R"("language_model.model.layers.0.mlp.shared_expert_gate": false)"},
           // A bf16 token table, which the gather reads as stored.
           {dense, object, unquantizedTable},
           {dense, object, R"("quantization": {
    "language_model.model.layers.0.linear_attn.in_proj_a": false,
    "language_model.model.layers.0.linear_attn.in_proj_b": false,)"},
       })
    static_cast<void>(inspect(accepted.model.with(&SourceModel::config, accepted.from, accepted.to)));
  struct Refused final {
    const SourceModel &model;
    std::string_view from, to, error;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           // MLX's own object, not the quantization_config a transformers
           // checkpoint (GPTQ, AWQ, ...) states.
           {dense, R"("quantization": {)", R"("unused": {)",
            "this model requires an MLX checkpoint (affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128, mxfp4 "
            "or nvfp4), an NVFP4 checkpoint of Model Optimizer or compressed-tensors, or a supported GGUF"},
           {dense, R"("bits": 4)", R"("bits": 7)",
            "quantization is affine 7-bit in groups of 64; MLX weights load as affine 2, 3, 4, 5, 6 or 8 bits in "
            "groups of 32, 64 or 128, or as mxfp4 or nvfp4"},
           {dense, R"("group_size": 64)", R"("group_size": 16)", "quantization is affine 4-bit in groups of 16"},
           // A group past a byte is no format (4 bits in groups of 320 are not af5g64).
           {dense, R"("group_size": 64)", R"("group_size": 320)", "quantization is affine 4-bit in groups of 320"},
           {dense, R"("mode": "affine")", R"("mode": "nvfp4")", "quantization is nvfp4 4-bit in groups of 64"},
           {moe, R"("bits": 8
    },)", R"("bits": 8, "mode": "mxfp8"
    },)", "quantization language_model.model.layers.0.mlp.gate is mxfp8 8-bit in groups of 64"},
       })
    refuses(refused.model.with(&SourceModel::config, refused.from, refused.to), refused.error,
            "an MLX quantization was accepted with " + std::string(refused.to));
  // A module the images read only quantized, left unquantized: a projection
  // of each kind and the head.
  struct Unquantized final {
    const SourceModel &model;
    std::string module;
  };
  for (const Unquantized &unquantized : std::initializer_list<Unquantized>{
           {dense, "language_model.model.layers.3.self_attn.q_proj"},
           {dense, "language_model.model.layers.0.linear_attn.in_proj_qkv"},
           {dense, "language_model.model.layers.0.mlp.up_proj"},
           {dense, "language_model.lm_head"},
           {moe, "language_model.model.layers.0.mlp.switch_mlp.down_proj"},
           {moe, "language_model.model.layers.0.mlp.shared_expert.gate_proj"},
       })
    refuses(unquantized.model.with(&SourceModel::config, object,
                                   std::string(object) + "\"" + unquantized.module + "\": false,"),
            "quantization " + unquantized.module +
                " is unquantized; Splash loads quantized MLX projections",
            "an unquantized " + unquantized.module + " was accepted");
  // Model Optimizer, as NVIDIA's releases state it: NVFP4 or FP8 for every
  // layer its ignore patterns leave, or each layer's own by quantized_layers,
  // a MoE layer's routed experts the one layer mlp.experts. Another algorithm
  // or group size, and a projection, the head or an expert's projection left
  // out of the layers or named by an ignore pattern, are refused.
  const auto modelOptimizer = [&](const SourceModel &model, std::string_view quantization) {
    return withQuantizationConfig(model, R"({"quant_method": "modelopt", )" + std::string(quantization) + "}");
  };
  // The MoE fixture's quantized_layers but `leftOut`: its 40 layers, each
  // fourth full attention.
  const auto moeLayers = [](std::string_view leftOut) {
    std::string layers = R"("quantized_layers": {"lm_head": {"quant_algo": "FP8"})";
    for (int layer = 0; layer < 40; ++layer) {
      const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
      std::vector<std::string> modules{"mlp.experts", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
                                       "mlp.shared_expert.down_proj"};
      if (layer % 4 == 3)
        modules.insert(modules.end(), {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"});
      else
        modules.insert(modules.end(), {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"});
      for (const std::string &module : modules)
        if (prefix + module != leftOut)
          layers += R"(, ")" + prefix + module + R"(": {"quant_algo": "W4A16_NVFP4", "group_size": 16})";
    }
    return R"("quant_algo": "MIXED_PRECISION", "ignore": ["mtp*"], )" + layers + "}";
  };
  for (const auto &[model, accepted] : std::initializer_list<std::pair<const SourceModel &, std::string>>{
           {dense, R"("quant_algo": "NVFP4", "group_size": 16, "ignore": ["mtp*", "model.visual*"])"},
           {dense, R"("quant_algo": "W4A16_NVFP4", "group_size": 16)"},
           {dense, R"("quant_algo": "FP8")"},
           {moe, moeLayers("")},
           {moe, R"("quant_algo": "NVFP4", "ignore": ["mtp*", "*mlp.gate", "*shared_expert_gate"])"},
       })
    static_cast<void>(inspect(modelOptimizer(model, accepted)));
  struct RefusedModelOptimizer final {
    const SourceModel &model;
    std::string quantization;
    std::string_view error;
  };
  for (const RefusedModelOptimizer &refused : std::initializer_list<RefusedModelOptimizer>{
           {dense, R"("quant_algo": "W4A8_AWQ")",
            "quantization_config is W4A8_AWQ; Model Optimizer weights load as NVFP4, W4A16_NVFP4 or FP8"},
           {dense, R"("quant_algo": "NVFP4", "group_size": 32)", "quantization_config is NVFP4 in groups of 32"},
           {dense, R"("quant_algo": "NVFP4", "ignore": ["*.self_attn.q_proj"])",
            "quantization_config leaves model.language_model.layers.3.self_attn.q_proj unquantized"},
           {dense, R"("quant_algo": "MIXED_PRECISION", "quantized_layers": {"lm_head": {"quant_algo": "NVFP4"}})",
            "quantization_config leaves model.language_model.layers.0.linear_attn.in_proj_qkv unquantized"},
           {dense, R"("quant_algo": "MIXED_PRECISION", "quantized_layers": {"lm_head": {"quant_algo": "INT4_AWQ"}})",
            "quantization_config quantized_layers lm_head is INT4_AWQ"},
           {moe, moeLayers("model.language_model.layers.0.mlp.experts"),
            "quantization_config leaves model.language_model.layers.0.mlp.experts unquantized"},
           {moe, R"("quant_algo": "NVFP4", "ignore": ["*.experts.*.down_proj"])",
            "quantization_config leaves model.language_model.layers.0.mlp.experts.0.down_proj unquantized"},
       })
    refuses(modelOptimizer(refused.model, refused.quantization), refused.error,
            "a Model Optimizer quantization was accepted with " + refused.quantization.substr(0, 200));
  // compressed-tensors, as unsloth's NVFP4 releases state it: FP8 per channel
  // for the attention and GDN projections and the head, NVFP4 for the FFN and
  // the experts, the last layers' experts FP8; or every projection NVFP4 by
  // class. A group of another format, ordered by activation groups or with a
  // target that is no regular expression, a pattern past 512 bytes or too
  // complex to match, and a projection (the first's or the last expert's) no
  // group targets or an ignore entry names, are refused.
  const auto compressedTensors = [&](const SourceModel &model, std::string_view groups, std::string_view ignore) {
    return withQuantizationConfig(
        model,
        std::string(R"({"quant_method": "compressed-tensors", "format": "mixed-precision", "config_groups": {)") +
            std::string(groups) + R"(}, "ignore": )" + std::string(ignore) + "}");
  };
  const std::string ignore = R"(["re:^mtp.*", "model.language_model.layers.0.linear_attn.in_proj_a"])";
  const std::string fp8Group =
      R"("group_0": {"format": "float-quantized", "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$", )"
      R"("re:.*linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)$", "re:.*lm_head", )"
      R"("re:.*layers\\.(38|39)\\.mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"], )"
      R"("weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": true}})";
  const auto nvfp4Group = [](std::string_view targets, std::string_view weights = R"("group_size": 16)") {
    return std::string(R"("group_1": {"format": "nvfp4-pack-quantized", "targets": )") + std::string(targets) +
           R"(, "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", )" + std::string(weights) +
           "}}";
  };
  const std::string ffn = nvfp4Group(R"(["re:.*mlp\\.(gate|up|down)_proj$", )"
                                     R"("re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$", )"
                                     R"("re:.*shared_expert\\.(gate|up|down)_proj$"])",
                                     R"("group_size": 16, "symmetric": true, "actorder": "static")");
  static_cast<void>(inspect(compressedTensors(dense, fp8Group + ", " + ffn, ignore)));
  static_cast<void>(inspect(compressedTensors(moe, fp8Group + ", " + ffn, ignore)));
  static_cast<void>(inspect(compressedTensors(dense, nvfp4Group(R"(["Linear"])"), "[]")));
  struct RefusedGroups final {
    const SourceModel &model;
    std::string groups, ignore;
    std::string_view error;
  };
  for (const RefusedGroups &refused : std::initializer_list<RefusedGroups>{
           {dense, fp8Group + ", " + ffn, R"(["lm_head"])", "quantization_config leaves lm_head unquantized"},
           {moe,
            fp8Group + ", " +
                nvfp4Group(R"(["re:.*mlp\\.experts\\.\\d+\\.(gate|up)_proj$", )"
                           R"("re:.*shared_expert\\.(gate|up|down)_proj$"])"),
            ignore, "quantization_config leaves model.language_model.layers.0.mlp.experts.0.down_proj unquantized"},
           {dense,
            R"("group_1": {"format": "pack-quantized", "targets": ["Linear"], )"
            R"("weights": {"num_bits": 4, "type": "int", )"
            R"("strategy": "group", "group_size": 128}})",
            "[]",
            "quantization_config config_groups group_1 is pack-quantized: int 4-bit by group in groups of 128; "
            "compressed-tensors weights load as NVFP4"},
           {dense, nvfp4Group(R"(["Linear"])", R"("group_size": 32)"), "[]",
            "quantization_config config_groups group_1 is nvfp4-pack-quantized: float 4-bit by tensor_group in "
            "groups of 32"},
           {dense, nvfp4Group(R"(["Linear"])", R"("group_size": 16, "actorder": "group")"), "[]",
            "quantization_config config_groups group_1 is asymmetric or ordered by activation groups"},
           {dense, nvfp4Group(R"(["re:(lm_head"])"), "[]",
            "quantization_config config_groups group_1 targets entry re:(lm_head is not a regular expression Splash "
            "reads"},
           {moe, fp8Group + ", " + ffn, R"(["re:.*mlp\\.experts\\.255\\.down_proj$"])",
            "quantization_config leaves model.language_model.layers.0.mlp.experts.255.down_proj unquantized"},
           {dense, nvfp4Group(R"(["Linear"])"), R"(["re:)" + std::string(600, 'x') + R"("])",
            "quantization_config ignore entry exceeds 512 bytes"},
           {dense, nvfp4Group(R"(["Linear"])"), R"(["re:(.+)+x$"])",
            "quantization_config ignore holds a regular expression too complex to match "
            "model.language_model.layers.0.linear_attn.in_proj_qkv"},
       })
    refuses(compressedTensors(refused.model, refused.groups, refused.ignore), refused.error,
            "a compressed-tensors quantization was accepted with " + refused.groups);
}

// The vision tower the record names is the family's, over RGB patches of two
// frames, without deepstack layers; a text-only installation reads none.
void testVisionConfig(const std::filesystem::path &fixtures) {
  const SourceModel text = mlxModel(fixtures, "qwen3.8-27b");
  require(!inspect(text.with(&SourceModel::config, R"("vision_config")", R"("unused")")).hasVision(),
          "a text-only installation read the vision config");
  const SourceModel vision = text.with(&SourceModel::record, R"("none")", R"("safetensors")");
  struct Refused final {
    std::string_view from, to, error;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {R"("vision_config")", R"("unused")", "vision config must be an object"},
           {R"("out_hidden_size": 5120)", R"("out_hidden_size": 2048)",
            "vision config out_hidden_size mismatch"},
           {R"("in_channels": 3)", R"("in_channels": 4)", "vision config in_channels mismatch"},
           {R"("hidden_act": "gelu_pytorch_tanh")", R"("hidden_act": "gelu")", "vision config hidden_act mismatch"},
           {R"("deepstack_visual_indexes": [])", R"("deepstack_visual_indexes": [8])",
            "vision deepstack layers are unsupported"},
       })
    refuses(vision.with(&SourceModel::config, refused.from, refused.to), refused.error,
            "a vision config was accepted with " + std::string(refused.to));
}

// The installer's check before any weight download (the engine's
// model-check): the family of a target's configuration, alone while its
// draft is not chosen and then with the draft's, by the rules a start
// applies. It refuses the fine-tunes an installer's narrower copy of those
// rules let through.
void testConfigurationCheck(const std::filesystem::path &fixtures) {
  const test::TemporaryDirectory directory("splash-model-check");
  const auto check = [&](std::string_view targetFormat, std::string_view visionFormat, std::string_view config,
                         std::optional<std::string_view> ggufMetadata, std::optional<std::string_view> draft) {
    test::writeFile(directory.path() / "config.json", config);
    std::optional<std::filesystem::path> metadataFile, draftConfig;
    if (ggufMetadata) test::writeFile(metadataFile.emplace(directory.path() / "gguf-metadata.json"), *ggufMetadata);
    if (draft) test::writeFile(draftConfig.emplace(directory.path() / "draft.json"), *draft);
    return std::string(model::inspectSourceConfiguration(targetFormat, visionFormat, directory.path() / "config.json",
                                                         metadataFile, draftConfig));
  };
  const SourceModel dense = mlxModel(fixtures, "qwen3.8-27b");
  const SourceModel moe = mlxModel(fixtures, "qwen3.6-35b-a3b");
  require(check("mlx-affine", "safetensors", dense.config, std::nullopt, dense.draft) == "Qwen3.8-27B" &&
              check("mlx-affine", "safetensors", moe.config, std::nullopt, moe.draft) == "Qwen3.6-35B-A3B",
          "a family's configuration was not found to be its family");
  // Before its draft is chosen, the target alone; then the draft must be the
  // family's.
  require(check("mlx-affine", "none", moe.config, std::nullopt, std::nullopt) == "Qwen3.6-35B-A3B",
          "a target without its draft was refused");
  rejects([&] { static_cast<void>(check("mlx-affine", "none", moe.config, std::nullopt, dense.draft)); },
          "draft config num_hidden_layers mismatch: checkpoint 5, runtime 6", "another family's draft was accepted");
  // A GGUF's config with the metadata the installer copies from its header,
  // which the planner checks in the GGUF at every start: what the config
  // does not hold, such as the RoPE scaling, the epsilon, the FFN widths and
  // the experts each token takes.
  const std::string denseMetadata = text(fixtures / "qwen3.8-27b" / "gguf-metadata.json");
  const std::string moeMetadata = text(fixtures / "qwen3.6-35b-a3b" / "gguf-metadata.json");
  require(check("gguf", "none", kDenseGgufConfig, denseMetadata, dense.draft) == "Qwen3.8-27B" &&
              check("gguf", "none", kMoeGgufConfig, moeMetadata, std::nullopt) == "Qwen3.6-35B-A3B",
          "a GGUF's derived config and metadata were not found to be its family");
  rejects([&] { static_cast<void>(check("gguf", "none", kDenseGgufConfig, std::nullopt, std::nullopt)); },
          "a GGUF target is checked with its metadata", "a GGUF was checked without its metadata");
  const auto replaced = [](std::string metadata, std::string_view from, std::string_view to) {
    const size_t at = metadata.find(from);
    require(at != std::string::npos, "the GGUF metadata has no " + std::string(from));
    return metadata.replace(at, from.size(), to);
  };
  // Qwen3.8-Flash-Next loads from a GGUF only and decodes without a draft;
  // its metadata also states the hyper-connections, the QSA indexer and the
  // PLE n-gram embedding.
  const std::string flashConfig = text(fixtures / "qwen3.8-flash-next" / "config.json");
  const std::string flashMetadata = text(fixtures / "qwen3.8-flash-next" / "gguf-metadata.json");
  require(check("gguf", "none", flashConfig, flashMetadata, std::nullopt) == "Qwen3.8-Flash-Next",
          "Qwen3.8-Flash-Next's GGUF config and metadata were not found to be its family");
  rejects([&] { static_cast<void>(check("gguf", "none", flashConfig, flashMetadata, dense.draft)); },
          "this model decodes without a DFlash2 draft, but a draft config was given",
          "a family without a draft accepted one");
  rejects([&] { static_cast<void>(check("mlx-affine", "none", flashConfig, std::nullopt, std::nullopt)); },
          "Qwen3.8-Flash-Next loads from a GGUF only", "Qwen3.8-Flash-Next was accepted from MLX");
  for (const auto &[from, to, error] : std::initializer_list<std::array<std::string_view, 3>>{
           {R"("qwen4exp.hyper_connection.count": 4)", R"("qwen4exp.hyper_connection.count": 2)",
            "GGUF metadata does not match the target: hyper_connection.count 2 (expected 4)"},
           {R"("qwen4exp.attention.indexer.top_k": 2048)", R"("qwen4exp.attention.indexer.top_k": 1024)",
            "GGUF metadata does not match the target: attention.indexer.top_k 1024 (expected 2048)"}}) {
    std::string metadata = flashMetadata;
    const size_t at = metadata.find(from);
    require(at != std::string::npos, "the GGUF metadata has no " + std::string(from));
    metadata.replace(at, from.size(), to);
    rejects([&] { static_cast<void>(check("gguf", "none", flashConfig, metadata, std::nullopt)); }, error,
            "Qwen3.8-Flash-Next accepted metadata the kernels do not compute");
  }
  struct RefusedMetadata final {
    std::string_view config;
    const std::string &metadata;
    std::string_view from, to, error;
  };
  for (const RefusedMetadata &refused : std::initializer_list<RefusedMetadata>{
           {kMoeGgufConfig, moeMetadata, R"("qwen35moe.expert_used_count": 8)", R"("qwen35moe.expert_used_count": 16)",
            "GGUF metadata does not match the target: expert_used_count 16 (expected 8)"},
           {kDenseGgufConfig, denseMetadata, R"("string": {)", R"("string": {"qwen35.rope.scaling.type": "yarn",)",
            "GGUF metadata does not match the target: rope.scaling.type yarn (expected none)"},
           {kDenseGgufConfig, denseMetadata, "9.999999974752427e-07", "1e-05",
            "GGUF metadata does not match the target: attention.layer_norm_rms_epsilon 1e-05 (expected 1e-06)"},
           {kDenseGgufConfig, denseMetadata, R"("qwen35.feed_forward_length": 17408)",
            R"("qwen35.feed_forward_length": 18944)",
            "GGUF metadata does not match the target: feed_forward_length 18944 (expected 17408)"},
           // Each value of its kind, an unsigned one a whole number to 2^64 - 1.
           {kDenseGgufConfig, denseMetadata, R"("qwen35.block_count": 65)", R"("qwen35.block_count": true)",
            "GGUF metadata qwen35.block_count must be a whole number from 0 to 2^64 - 1"},
           {kDenseGgufConfig, denseMetadata, R"("qwen35.block_count": 65)", R"("qwen35.block_count": -1)",
            "GGUF metadata qwen35.block_count must be a whole number from 0 to 2^64 - 1"},
           {kDenseGgufConfig, denseMetadata, R"("qwen35.block_count": 65)",
            R"("qwen35.block_count": 18446744073709551615)", "block_count 18446744073709551615 (expected 65)"},
       })
    rejects([&] {
      static_cast<void>(
          check("gguf", "none", refused.config, replaced(refused.metadata, refused.from, refused.to), std::nullopt));
    }, refused.error, "GGUF metadata was accepted with " + std::string(refused.to));
  struct Refused final {
    const SourceModel &model;
    std::string SourceModel::*file;
    std::string_view from, to, error;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {dense, &SourceModel::config, R"("rope_type": "default")", R"("rope_type": "yarn")",
            "text config rope_parameters rope_type mismatch: MLX yarn, runtime default"},
           {dense, &SourceModel::config, R"("tie_word_embeddings": false)", R"("tie_word_embeddings": true)",
            "text config tie_word_embeddings must be false"},
           {dense, &SourceModel::config, R"("rms_norm_eps": 1e-06)", R"("rms_norm_eps": 1e-05)",
            "text config rms_norm_eps mismatch: MLX 1e-05, runtime 1e-06"},
           // The JSON true equals 1, the first layer this draft reads, in Python.
           {moe, &SourceModel::draft, "[\n      1,", "[\n      true,",
            "draft config target_layer_ids 0 must be a number"},
       }) {
    const SourceModel changed = refused.model.with(refused.file, refused.from, refused.to);
    rejects([&] { static_cast<void>(check("mlx-affine", "safetensors", changed.config, std::nullopt, changed.draft)); },
            refused.error, "a fine-tune was accepted with " + std::string(refused.to));
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: model-configuration FIXTURES");
    testFamilies(argv[1]);
    testSources();
    testOneRulePerValue(argv[1]);
    testQuantization(argv[1]);
    testVisionConfig(argv[1]);
    testConfigurationCheck(argv[1]);
    std::cout << "model configuration: every family, sources, one rule per value, quantization, vision, the "
                 "installer's check PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
