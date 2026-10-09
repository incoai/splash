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
              denseDescriptor.targetSource == model::TargetSource::Mlx && !denseDescriptor.hasVision() &&
              denseDescriptor.draft == model::kQwen3_8DraftLayout,
          "an MLX Qwen3.8-27B made another descriptor");
  // The sources' identity is the digest of the record naming them.
  require(denseDescriptor.sourceIdentity == model::weightDigest(dense.record) &&
              inspect(dense.with(&SourceModel::record, "community/fine-tune", "community/other"))
                      .sourceIdentity != denseDescriptor.sourceIdentity,
          "the sources' identity is not the digest of the record naming them");
  require(inspect(dense.with(&SourceModel::record, R"("none")", R"("safetensors")")).visionSource ==
              model::VisionSource::Mlx,
          "an MLX Qwen3.8-27B with its vision tower made another descriptor");
  const SourceModel moe = mlxModel(fixtures, "qwen3.6-35b-a3b");
  const model::ModelDescriptor moeDescriptor =
      inspect(moe.with(&SourceModel::record, R"("none")", R"("safetensors")"));
  require(std::holds_alternative<model::Qwen3_6MoeLayout>(moeDescriptor.target) &&
              moeDescriptor.visionSource == model::VisionSource::Mlx &&
              moeDescriptor.draft == model::kQwen3_6MoeDraftLayout,
          "an MLX Qwen3.6-35B-A3B made another descriptor");
  // The model type names the family; the sizes every source's config states
  // tell its target from other models of the architecture, which Splash does
  // not serve, such as Qwen3.5-4B.
  constexpr std::string_view supported = "; supported: Qwen3.8-27B, Qwen3.6-35B-A3B";
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
  require(described(model::TargetSource::Mlx, model::VisionSource::None).valid(),
          "a descriptor of its sources was not valid");
  require(!described(model::TargetSource{}, model::VisionSource::None).valid() &&
              !described(model::TargetSource::Mlx, model::VisionSource{}).valid(),
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

// An MLX target's quantization is MLX's own object: the object and each
// module's own entry name affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64
// or 128, or mxfp4; an entry is affine unless it names its mode, and takes its
// mode's default bits and group size (affine 4 and 64, mxfp4 4 and 32) when it
// omits them, as MLX loads it. The entry false, which MLX writes for a module
// it leaves unquantized, is refused for a projection, the head and the token
// table, and accepted for the router, the shared-expert gate and GDN alpha and
// beta. A GGUF target has none (above).
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
  struct Accepted final {
    const SourceModel &model;
    std::string_view from, to;
  };
  for (const Accepted &accepted : std::initializer_list<Accepted>{
           {dense, R"("bits": 4)", R"("bits": 8)"},
           {dense, R"("bits": 4)", R"("bits": 3)"},
           {dense, R"("group_size": 64)", R"("group_size": 128)"},
           {dense, affine, mxfp4},
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
            "this model requires an MLX checkpoint (affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128, or "
            "mxfp4) or a supported GGUF"},
           {dense, R"("bits": 4)", R"("bits": 7)",
            "quantization is affine 7-bit in groups of 64; MLX weights load as affine 2, 3, 4, 5, 6 or 8 bits in "
            "groups of 32, 64 or 128, or as mxfp4"},
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
  // of each kind, the head and the token table.
  struct Unquantized final {
    const SourceModel &model;
    std::string module;
  };
  for (const Unquantized &unquantized : std::initializer_list<Unquantized>{
           {dense, "language_model.model.layers.3.self_attn.q_proj"},
           {dense, "language_model.model.layers.0.linear_attn.in_proj_qkv"},
           {dense, "language_model.model.layers.0.mlp.up_proj"},
           {dense, "language_model.lm_head"},
           {dense, "language_model.model.embed_tokens"},
           {moe, "language_model.model.layers.0.mlp.switch_mlp.down_proj"},
           {moe, "language_model.model.layers.0.mlp.shared_expert.gate_proj"},
       })
    refuses(unquantized.model.with(&SourceModel::config, object,
                                   std::string(object) + "\"" + unquantized.module + "\": false,"),
            "quantization " + unquantized.module +
                " is unquantized; Splash loads quantized MLX projections and token tables",
            "an unquantized " + unquantized.module + " was accepted");
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
    std::cout << "model configuration: both families, sources, one rule per value, quantization, vision, the "
                 "installer's check PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
