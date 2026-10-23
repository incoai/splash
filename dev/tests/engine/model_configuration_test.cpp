// The rules an upstream model's configuration is held to before any of its
// weights load (model/ModelDescriptor.mm), at every start and by the
// installer's check before any weight download, on configs built like each
// supported family's: the MLX target's config.json and its DFlash2 draft's,
// read from FIXTURES (dev/tests/fixtures/model-configs/<family>), and the
// config the installer derives from a GGUF's metadata with the metadata it
// copies (gguf-metadata.json there). Also the descriptor a context past the
// native window makes of one, and YaRN's rotary frequencies.
//
//   model-configuration FIXTURES

#include "TestChecks.hpp"
#include "TestDrafts.hpp"
#include "TestFiles.hpp"
#include "TestModel.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/RopeScaling.hpp"
#include "model/WeightStore.hpp"

#include <array>
#include <bit>
#include <cmath>
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
    R"("head_dim":256,"num_experts":256,"num_hidden_layers":40,"vocab_size":248320}})";
// Qwen3.5-122B-A10B's, of the same model type, from unsloth's
// Ornith-1.5-122B GGUF (its MTP layer left out of the layers).
constexpr std::string_view kLargeMoeGgufConfig =
    R"({"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":3072,)"
    R"("max_position_embeddings":262144,"num_attention_heads":32,"num_key_value_heads":2,)"
    R"("head_dim":256,"num_experts":256,"num_hidden_layers":48,"vocab_size":248320}})";

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
              denseDescriptor.draft == test::qwen3_8Draft(),
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
              moeDescriptor.draft == test::qwen3_6MoeDraft(),
          "an MLX Qwen3.6-35B-A3B made another descriptor");
  // Qwen3.5-122B-A10B's model type is the Qwen3.6-35B-A3B's: its sizes
  // name its family (describeSourceModel).
  const model::ModelDescriptor largeMoeDescriptor =
      inspect(mlxModel(fixtures, "qwen3.5-122b-a10b").with(&SourceModel::record, R"("none")", R"("safetensors")"));
  require(largeMoeDescriptor.target == model::TargetLayout{model::kQwen3_5_122BLayout} &&
              largeMoeDescriptor.visionSource == model::VisionSource::Safetensors &&
              largeMoeDescriptor.vision.outputHiddenSize == 3072 &&
              largeMoeDescriptor.draft == test::qwen3_5_122BDraft(),
          "an MLX Qwen3.5-122B-A10B made another descriptor");
  // Qwen3.8-Flash-Next's GGUF config, with its draft's.
  const model::ModelDescriptor flashDescriptor =
      inspect(mlxModel(fixtures, "qwen3.8-flash-next").with(&SourceModel::record, R"("mlx-affine")", R"("gguf")"));
  require(std::holds_alternative<model::Qwen4ExpLayout>(flashDescriptor.target) &&
              flashDescriptor.targetSource == model::TargetSource::Gguf &&
              flashDescriptor.draft == test::flashNextDraft(),
          "a GGUF Qwen3.8-Flash-Next made another descriptor");
  // Its KV pages also hold the QSA indexer's pooled keys and its states the
  // PLE convolution's history and the indexer's pending keys, which another
  // family's do not.
  const model::Qwen4ExpLayout flash;
  require(flashDescriptor.targetKvLayout.indexDimension == flash.indexerHeadDimension &&
              flashDescriptor.stateLayout.target.auxiliaryBytes ==
                  flash.pleHistoryBytes() + flash.indexerPendingBytes() &&
              flash.pleHistoryBytes() && flash.indexerPendingBytes() == 12 * 3 * 128 * 2 &&
              !denseDescriptor.targetKvLayout.indexDimension &&
              !denseDescriptor.stateLayout.target.auxiliaryBytes,
          "a family's cache layouts do not hold what it adds");
  // Its full-attention layers select blocks, so its K/V may spill: a verify
  // row attends 512 blocks of 4 tokens, and a row with at most 2,048 tokens
  // of history attends them through its page table, the shape the engine's
  // tests take (TestModel.hpp). Another family's rows attend every token. A
  // descriptor stating another shape than its target's is invalid.
  require(flashDescriptor.capabilities.sparseKv == test::sparseKvShape() &&
              !denseDescriptor.capabilities.sparseKv && !moeDescriptor.capabilities.sparseKv,
          "a family's descriptor does not state the sparse KV its attention reads");
  model::ModelDescriptor undeclared = flashDescriptor;
  undeclared.capabilities.sparseKv.reset();
  require(flashDescriptor.valid() && !undeclared.valid(),
          "a descriptor stating another sparse KV than its target's was valid");
  // The attention kernels read blocks of 4 tokens: a target whose indexer
  // pools other blocks is invalid, whatever it states.
  model::ModelDescriptor otherBlocks = flashDescriptor;
  std::get<model::Qwen4ExpLayout>(otherBlocks.target).indexerBlockTokens = 8;
  otherBlocks.capabilities.sparseKv->blockTokens = 8;
  otherBlocks.capabilities.sparseKv->denseHistoryTokens = 4'096;
  require(!otherBlocks.valid(), "a target whose blocks are not the attention kernels' was valid");
  // A draft states its own layers, intermediate size, rotary base and the
  // target layers it reads.
  constexpr std::array<uint32_t, 6> ownLayers{5, 19, 33, 47, 61, 63};
  require(inspect(dense.with(&SourceModel::draft, R"("num_hidden_layers": 5)", R"("num_hidden_layers": 3)")
                      .with(&SourceModel::draft, R"("intermediate_size": 17408)", R"("intermediate_size": 8192)")
                      .with(&SourceModel::draft, R"("rope_theta": 10000000)", R"("rope_theta": 1000000)")
                      .with(&SourceModel::draft, "61\n", "61,\n      63\n"))
                  .draft == model::draftLayout(5120, 248320, 3, 8192, 1e6F, ownLayers),
          "a draft's own sizes were not its descriptor's");
  // The model type names the family; the sizes every source's config states
  // tell its target from other models of the architecture, which Splash does
  // not serve, such as Qwen3.5-4B.
  constexpr std::string_view supported =
      "; supported: Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B, Qwen3.8-Flash-Next";
  struct Refused final {
    std::string_view from, to, difference;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {R"("hidden_size": 5120)", R"("hidden_size": 2560)", "text config hidden_size: safetensors 2560, Qwen3.8-27B 5120"},
           {R"("num_hidden_layers": 64)", R"("num_hidden_layers": 32)",
            "text config num_hidden_layers: safetensors 32, Qwen3.8-27B 64"},
           {R"("model_type": "qwen3_5_text")", R"("model_type": "gemma3_text")", "text model type gemma3_text"},
           {R"("text_config")", R"("unused")", "its config has no text_config"},
       })
    refuses(dense.with(&SourceModel::config, refused.from, refused.to),
            "no supported model has this architecture (" + std::string(refused.difference) + ")" +
                std::string(supported),
            "a config of another model was accepted with " + std::string(refused.to));
  // A config of a model type of several families and the sizes of none is
  // refused as it differs from the one it matches furthest.
  refuses(mlxModel(fixtures, "qwen3.5-122b-a10b")
              .with(&SourceModel::config, R"("num_hidden_layers": 48)", R"("num_hidden_layers": 40)"),
          "no supported model has this architecture (text config num_hidden_layers: safetensors 40, "
          "Qwen3.5-122B-A10B 48)" +
              std::string(supported),
          "a config of no family's sizes was accepted");
  // One that differs from every family in the first size, the hidden size,
  // matches none further than the first family of its model type, and is
  // refused as it differs from that one.
  refuses(moe.with(&SourceModel::config, R"("hidden_size": 2048)", R"("hidden_size": 4096)"),
          "no supported model has this architecture (text config hidden_size: safetensors 4096, "
          "Qwen3.6-35B-A3B 2048)" +
              std::string(supported),
          "a config of no family's hidden size was accepted");
}

// A descriptor states where its weights come from: one that leaves either
// source unset, which hasVision would read as a vision tower, is not valid;
// nor is one whose draft reads past the target's last layer.
void testSources() {
  const auto described = [](model::TargetSource target, model::VisionSource vision,
                            const model::DFlashDraftLayout &draft = test::qwen3_8Draft()) {
    return model::makeModelDescriptor("sources", model::Qwen3_8Layout{}, draft, test::qwen3_8Vision(), target,
                                      vision);
  };
  require(described(model::TargetSource::Safetensors, model::VisionSource::None).valid(),
          "a descriptor of its sources was not valid");
  require(!described(model::TargetSource{}, model::VisionSource::None).valid() &&
              !described(model::TargetSource::Safetensors, model::VisionSource{}).valid(),
          "a descriptor without a source was valid");
  constexpr std::array<uint32_t, 2> pastLastLayer{5, 64};
  require(!described(model::TargetSource::Safetensors, model::VisionSource::None,
                     model::draftLayout(5120, 248320, 5, 17408, 1e7F, pastLastLayer))
               .valid(),
          "a descriptor whose draft reads past the target's last layer was valid");
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
            "text config linear_num_value_heads mismatch: safetensors 47, runtime 48"},
           {&SourceModel::config, R"("attn_output_gate": true)", R"("attn_output_gate": 1)",
            "text config attn_output_gate must be true"},
           {&SourceModel::config, R"("rope_type": "default")", R"("type": "yarn")",
            "text config rope_parameters type mismatch"},
           {&SourceModel::config, R"("full_attention")", R"("linear_attention")",
            "text config layer_types 3 must be full_attention"},
           {&SourceModel::draft, "[\n      5,", "[\n      5.9,",
            "draft config target_layer_ids 0 must be a whole number"},
           {&SourceModel::draft, "61\n", "61,\n      1\n", "a DFlash2 draft reads 1 to 8 target layers, ascending"},
           {&SourceModel::draft, "61\n", "61,\n      64\n",
            "draft config target_layer_ids names a layer past the target's"},
           {&SourceModel::draft, R"("intermediate_size": 17408)", R"("intermediate_size": 17416)",
            "a DFlash2 draft's intermediate size is a positive multiple of 64"},
           {&SourceModel::draft, R"("num_hidden_layers": 5)", R"("num_hidden_layers": 0)",
            "a DFlash2 draft has at least one layer"},
           {&SourceModel::draft, R"("rope_theta": 10000000)", R"("rope_theta": -1)",
            "a DFlash2 draft's rotary theta is positive"},
           {&SourceModel::draft, R"("num_attention_heads": 32)", R"("num_attention_heads": 16)",
            "draft config num_attention_heads mismatch: checkpoint 16, runtime 32"},
           {&SourceModel::draft, R"("selector_top_k": 16)", R"("selector_top_k": 8)",
            "draft config dflash_config selector_top_k mismatch"},
           {&SourceModel::draft, R"("selector_top_k": 16)",
            R"("selector_top_k": 16, "target_hidden_reduction": "stream_mean")",
            "draft config dflash_config target_hidden_reduction mismatch: checkpoint stream_mean, runtime none"},
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
            "or nvfp4), an NVFP4 or FP8 checkpoint of Model Optimizer or compressed-tensors, or a supported GGUF"},
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

// A GGUF MoE target is served by the routed experts its config states, which
// the installer derives from the header, fewer than its family's in a pruned
// release: a multiple of 8 from the experts each token takes to the family's
// (ggufRoutedExperts). Every other dimension stays the family's, and a config
// without the count, which an earlier release derived, states the family's.
// An MLX target's count is the family's.
void testPrunedExperts(const std::filesystem::path &fixtures) {
  // The family's layout with `experts` routed experts.
  const auto withExperts = [](auto layout, uint32_t experts) -> model::TargetLayout {
    layout.experts = experts;
    return layout;
  };
  const SourceModel flash =
      mlxModel(fixtures, "qwen3.8-flash-next").with(&SourceModel::record, R"("mlx-affine")", R"("gguf")");
  const auto flashWith = [&](uint32_t experts) {
    return flash.with(&SourceModel::config, R"("num_experts": 512)",
                      R"("num_experts": )" + std::to_string(experts));
  };
  const model::ModelDescriptor pruned = inspect(flashWith(256));
  require(pruned.valid() && pruned.target == withExperts(model::Qwen4ExpLayout{}, 256) &&
              pruned.draft == test::flashNextDraft(),
          "a pruned Qwen3.8-Flash-Next GGUF was not described as its family with its experts");
  for (const uint32_t experts : {16U, 512U})
    require(inspect(flashWith(experts)).target == withExperts(model::Qwen4ExpLayout{}, experts),
            "a Qwen3.8-Flash-Next GGUF of " + std::to_string(experts) + " experts was not served by them");
  require(inspect(flash.with(&SourceModel::config, R"("num_experts": 512,)", "")).target ==
              model::TargetLayout{model::Qwen4ExpLayout{}},
          "a GGUF config without its experts did not state the family's");
  for (const uint32_t experts : {8U, 12U, 520U})
    refuses(flashWith(experts),
            "(text config num_experts: GGUF " + std::to_string(experts) +
                ", Qwen3.8-Flash-Next a multiple of 8 from 16 to 512)",
            "a Qwen3.8-Flash-Next GGUF of " + std::to_string(experts) + " experts was accepted");
  refuses(flash.with(&SourceModel::config, R"("num_experts": 512)", R"("num_experts": true)"),
          "text config num_experts must be a number", "a GGUF config's experts were read from a boolean");
  // Qwen3.6-35B-A3B's, by its own experts and the experts each token takes.
  SourceModel moe = mlxModel(fixtures, "qwen3.6-35b-a3b").with(&SourceModel::record, R"("mlx-affine")", R"("gguf")");
  moe.config = kMoeGgufConfig;
  const auto moeWith = [&](uint32_t experts) {
    return moe.with(&SourceModel::config, R"("num_experts":256)", R"("num_experts":)" + std::to_string(experts));
  };
  for (const uint32_t experts : {8U, 128U})
    require(inspect(moeWith(experts)).target == withExperts(model::Qwen3_6MoeLayout{}, experts),
            "a Qwen3.6-35B-A3B GGUF of " + std::to_string(experts) + " experts was not served by them");
  refuses(moeWith(264), "(text config num_experts: GGUF 264, Qwen3.6-35B-A3B a multiple of 8 from 8 to 256)",
          "a Qwen3.6-35B-A3B GGUF of more experts than its family's was accepted");
  refuses(mlxModel(fixtures, "qwen3.6-35b-a3b")
              .with(&SourceModel::config, R"("num_experts": 256)", R"("num_experts": 128)"),
          "text config num_experts mismatch: safetensors 128, runtime 256",
          "an MLX target of other experts than its family's was accepted");
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
          "draft config hidden_size mismatch: checkpoint 5120, runtime 2048", "another family's draft was accepted");
  // A GGUF's config with the metadata the installer copies from its header,
  // which the planner checks in the GGUF at every start: what the config
  // does not hold, such as the RoPE scaling, the epsilon, the FFN widths and
  // the experts each token takes.
  const std::string denseMetadata = text(fixtures / "qwen3.8-27b" / "gguf-metadata.json");
  const std::string moeMetadata = text(fixtures / "qwen3.6-35b-a3b" / "gguf-metadata.json");
  require(check("gguf", "none", kDenseGgufConfig, denseMetadata, dense.draft) == "Qwen3.8-27B" &&
              check("gguf", "none", kMoeGgufConfig, moeMetadata, std::nullopt) == "Qwen3.6-35B-A3B",
          "a GGUF's derived config and metadata were not found to be its family");
  // Qwen3.5-122B-A10B, of the same model type and GGUF architecture as the
  // Qwen3.6-35B-A3B, as its sizes tell; its draft, trained to fill a block
  // of 16, drafts the kernels' eight rows, and a block of fewer is refused.
  // Each family refuses the other's draft.
  const SourceModel largeMoe = mlxModel(fixtures, "qwen3.5-122b-a10b");
  const std::string largeMoeMetadata = text(fixtures / "qwen3.5-122b-a10b" / "gguf-metadata.json");
  require(check("mlx-affine", "safetensors", largeMoe.config, std::nullopt, largeMoe.draft) == "Qwen3.5-122B-A10B" &&
              check("gguf", "none", kLargeMoeGgufConfig, largeMoeMetadata, largeMoe.draft) == "Qwen3.5-122B-A10B",
          "a Qwen3.5-122B-A10B's configuration was not found to be its family");
  rejects([&] { static_cast<void>(check("mlx-affine", "none", moe.config, std::nullopt, largeMoe.draft)); },
          "draft config hidden_size mismatch: checkpoint 3072, runtime 2048", "a larger family's draft was accepted");
  rejects([&] { static_cast<void>(check("mlx-affine", "none", largeMoe.config, std::nullopt, moe.draft)); },
          "draft config hidden_size mismatch: checkpoint 2048, runtime 3072", "a smaller family's draft was accepted");
  rejects([&] {
    static_cast<void>(check("mlx-affine", "none", largeMoe.config, std::nullopt,
                            largeMoe.with(&SourceModel::draft, R"("block_size": 16)", R"("block_size": 4)").draft));
  }, "draft config dflash_config block_size is less than the 8 rows a cycle drafts",
          "a draft of a block shorter than the kernels' rows was accepted");
  rejects([&] { static_cast<void>(check("gguf", "none", kDenseGgufConfig, std::nullopt, std::nullopt)); },
          "a GGUF target is checked with its metadata", "a GGUF was checked without its metadata");
  const auto replaced = [](std::string json, std::string_view from, std::string_view to) {
    const size_t at = json.find(from);
    require(at != std::string::npos, "the config or metadata has no " + std::string(from));
    return json.replace(at, from.size(), to);
  };
  // Qwen3.8-Flash-Next loads from a GGUF only; its metadata also states the
  // hyper-connections, the QSA indexer and the PLE n-gram embedding.
  const std::string flashConfig = text(fixtures / "qwen3.8-flash-next" / "config.json");
  const std::string flashMetadata = text(fixtures / "qwen3.8-flash-next" / "gguf-metadata.json");
  const std::string flashDraft = text(fixtures / "qwen3.8-flash-next" / "draft" / "config.json");
  require(check("gguf", "none", flashConfig, flashMetadata, std::nullopt) == "Qwen3.8-Flash-Next" &&
              check("gguf", "none", flashConfig, flashMetadata, flashDraft) == "Qwen3.8-Flash-Next",
          "Qwen3.8-Flash-Next's GGUF config and metadata were not found to be its family");
  rejects([&] { static_cast<void>(check("gguf", "none", flashConfig, flashMetadata, dense.draft)); },
          "draft config hidden_size mismatch: checkpoint 5120, runtime 2560", "another family's draft was accepted");
  // Its draft reads the mean of a layer's streams, which a config states or
  // leaves to the target.
  const auto flashDraftWith = [&](std::string_view to) {
    std::string draft = flashDraft;
    const std::string_view from = R"("target_hidden_reduction": "stream_mean",)";
    return draft.replace(draft.find(from), from.size(), to);
  };
  require(check("gguf", "none", flashConfig, flashMetadata, flashDraftWith("")) == "Qwen3.8-Flash-Next",
          "a Qwen3.8-Flash-Next draft without target_hidden_reduction was refused");
  const std::string otherReduction = flashDraftWith(R"("target_hidden_reduction": "none",)");
  rejects([&] { static_cast<void>(check("gguf", "none", flashConfig, flashMetadata, otherReduction)); },
          "draft config dflash_config target_hidden_reduction mismatch: checkpoint none, runtime stream_mean",
          "a draft of another reduction was accepted");
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
  // A pruned release, such as its Coder GGUF of 256 of the 512 routed
  // experts: the config derived from the header states the header's count,
  // which the header holds it to, and every other value stays the family's.
  const std::string coderConfig = replaced(flashConfig, R"("num_experts": 512)", R"("num_experts": 256)");
  const std::string coderMetadata =
      replaced(flashMetadata, R"("qwen4exp.expert_count": 512)", R"("qwen4exp.expert_count": 256)");
  require(check("gguf", "none", coderConfig, coderMetadata, std::nullopt) == "Qwen3.8-Flash-Next" &&
              check("gguf", "none", coderConfig, coderMetadata, flashDraft) == "Qwen3.8-Flash-Next",
          "a pruned Qwen3.8-Flash-Next GGUF's config and metadata were not found to be its family");
  rejects([&] { static_cast<void>(check("gguf", "none", coderConfig, flashMetadata, std::nullopt)); },
          "GGUF metadata does not match the target: expert_count 512 (expected 256)",
          "a config of other experts than its header's was accepted");
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
            "text config rope_parameters rope_type mismatch: safetensors yarn, runtime default"},
           {dense, &SourceModel::config, R"("tie_word_embeddings": false)", R"("tie_word_embeddings": true)",
            "text config tie_word_embeddings must be false"},
           {dense, &SourceModel::config, R"("rms_norm_eps": 1e-06)", R"("rms_norm_eps": 1e-05)",
            "text config rms_norm_eps mismatch: safetensors 1e-05, runtime 1e-06"},
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

// A family served for a context past its 262,144-token window takes YaRN by
// the context over the native window (servedFor), the factor Qwen gives for
// a context, and that context as its window, up to the 1,010,000 tokens Qwen
// validates; within the native window, and for auto, it is served as it is.
// A family that states no YaRN window serves no context past its native
// one. YaRN's rotary frequencies follow its reference computation
// (beta_fast 32, beta_slow 1), here for the targets' 32 pairs of base 10^7,
// whose correction range runs from pair 14 to pair 22: below it each
// frequency is kept, above it divided by the factor, between them blended.
void testYarn(const std::filesystem::path &fixtures) {
  for (const SourceModel &source :
       {mlxModel(fixtures, "qwen3.8-27b"), mlxModel(fixtures, "qwen3.6-35b-a3b"),
        mlxModel(fixtures, "qwen3.8-flash-next").with(&SourceModel::record, R"("mlx-affine")", R"("gguf")")}) {
    const model::ModelDescriptor native = inspect(source);
    require(native.capabilities.maximumContextTokens == 262144 && !native.ropeScaling.yarn(),
            "a model served without YaRN has another window or a RoPE scaling");
    for (const uint32_t context : {0U, 1U, 262144U}) {
      const model::ModelDescriptor served = model::servedFor(native, context);
      require(served.valid() && served.capabilities.maximumContextTokens == 262144 && !served.ropeScaling.yarn(),
              "a context of " + std::to_string(context) + " tokens was not served by the native window");
    }
    for (const auto [context, factor] : {std::pair{262145U, 1.000003814697265625}, {300000U, 1.1444091796875},
                                         {524288U, 2.0}, {1000000U, 3.814697265625}, {1010000U, 3.85284423828125}}) {
      const model::ModelDescriptor scaled = model::servedFor(native, context);
      require(scaled.valid() && scaled.capabilities.maximumContextTokens == context &&
                  scaled.ropeScaling == model::RopeScaling{factor, 262144},
              "a context of " + std::to_string(context) + " tokens was not served with YaRN by its ratio to the "
              "native window");
      rejects([&] { static_cast<void>(model::servedFor(scaled, context)); }, "already scaled",
              "a model's RoPE was scaled twice");
    }
    rejects([&] { static_cast<void>(model::servedFor(native, 1010001)); }, "serves at most 1010000 tokens",
            "a context past YaRN's window was served");
    // A model is valid only with the window its scaling serves, and within
    // YaRN's: by 4 that is 1,048,576 tokens, past the 1,010,000.
    model::ModelDescriptor beyond = native;
    beyond.ropeScaling = {4.0, 262144};
    for (const uint32_t window : {1010000U, 1048576U}) {
      beyond.capabilities.maximumContextTokens = window;
      require(!beyond.valid(), "a model scaled by 4 was valid with a window of " + std::to_string(window) + " tokens");
    }
    // A context over a native window it does not divide, whose factor is
    // inexact, is served whole.
    model::ModelDescriptor other = native;
    std::visit([](auto &layout) { layout.nativeContextTokens = 100000; }, other.target);
    other.capabilities.maximumContextTokens = 100000;
    require(model::servedFor(other, 100002).capabilities.maximumContextTokens == 100002,
            "a context over another native window was not served whole");
    model::ModelDescriptor unserved = native;
    std::visit([](auto &layout) { layout.yarnContextTokens = 0; }, unserved.target);
    rejects([&] { static_cast<void>(model::servedFor(unserved, 262145)); }, "serves at most 262144 tokens",
            "a family without a YaRN window was served with YaRN");
  }

  constexpr uint32_t kPairs = 32;
  constexpr float kTheta = 1.0e7F;
  std::array<float, kPairs> plain{}, scaled{};
  model::rotaryInverseFrequencies(kTheta, kPairs, {}, plain);
  for (uint32_t pair = 0; pair < kPairs; ++pair)
    require(plain[pair] == std::pow(kTheta, -static_cast<float>(pair) / kPairs),
            "the frequencies without YaRN changed");
  const model::RopeScaling yarn{4.0, 262144};
  model::rotaryInverseFrequencies(kTheta, kPairs, yarn, scaled);
  for (uint32_t pair = 0; pair < kPairs; ++pair) {
    if (pair <= 14)
      require(scaled[pair] == plain[pair], "YaRN changed a frequency it keeps");
    else if (pair >= 22)
      require(scaled[pair] == plain[pair] / 4.0F, "YaRN did not divide a frequency it interpolates");
  }
  // The reference computation's float32 values of the blended pairs.
  for (const auto [pair, expected] : {std::pair{15U, 4.7423981595784426e-04}, {18U, 7.217386882985011e-05},
                                      {21U, 8.759770025790203e-06}})
    require(std::abs(scaled[pair] - expected) <= 1e-6 * expected,
            "YaRN blended pair " + std::to_string(pair) + " to another frequency");
  require(yarn.attentionFactor() == 1.13862943649292F && model::RopeScaling{}.attentionFactor() == 1.0F,
          "YaRN's attention factor is not 0.1 ln(factor) + 1");
}

// Flint's configs are flat, its GGUF config the release's config.json or
// one the installer derives with the release's keys of it: the layout's
// values hold each key to it by name, and its draft names the mask token,
// one of Flint's reserved tokens, which the descriptor's layout then holds.
void testFlint(const std::filesystem::path &fixtures) {
  const SourceModel flint =
      mlxModel(fixtures, "flint").with(&SourceModel::record, R"("mlx-affine")", R"("gguf")");
  const model::ModelDescriptor descriptor = inspect(flint);
  const auto *layout = std::get_if<model::FlintLayout>(&descriptor.target);
  require(layout && descriptor.targetSource == model::TargetSource::Gguf && descriptor.family() == "Flint" &&
              layout->maskToken == 150019 && descriptor.draft == test::flintDraft() && descriptor.valid(),
          "a GGUF Flint made another descriptor");
  // Its 35 sliding layers read a window of the 512 keys before a row's, its
  // 14 others every key, the 49 attention layers' K/V in one page.
  require(descriptor.targetKvLayout.attentionLayers == 49 && descriptor.targetKvLayout.windowTokens == 512 &&
              std::popcount(descriptor.targetKvLayout.windowedLayers) == 35 && !descriptor.stateLayout.target.layers,
          "Flint's KV and state layouts are not its layers'");
  const auto masked = [&](uint32_t token) {
    return flint.with(&SourceModel::draft, R"("mask_token_id": 150019)",
                      R"("mask_token_id": )" + std::to_string(token));
  };
  require(std::get<model::FlintLayout>(inspect(masked(150143)).target).maskToken == 150143,
          "a Flint draft's last reserved mask token was not its layout's");
  for (const uint32_t token : {150018u, 150144u, 248077u})
    refuses(masked(token), "is not one of the target's reserved tokens 150019 to 150143",
            "a Flint draft's mask token outside the reserved tokens was accepted");
  refuses(flint.with(&SourceModel::record, R"("gguf")", R"("mlx-affine")"), "Flint loads from a GGUF only",
          "an MLX Flint was accepted");
  // Each key Flint's values make is held to them by name.
  struct Refused final {
    std::string_view from, to, error;
  };
  for (const Refused &refused : std::initializer_list<Refused>{
           {R"("sliding_window": 513)", R"("sliding_window": 512)", "text config sliding_window mismatch"},
           {R"("sliding_attention")", R"("full_attention")", "text config layer_types 2 must be sliding_attention"},
           {"\n    48\n  ]", "\n    47\n  ]", "text config no_rope_layer_indices mismatch"},
           {R"("softmax_type": "learnable")", R"("softmax_type": "vanilla")", "text config softmax_type mismatch"},
           {R"("router_score_function": "sigmoid")", R"("router_score_function": "softmax")",
            "text config router_score_function mismatch"},
           {R"("norm_topk_prob": true)", R"("norm_topk_prob": false)", "text config norm_topk_prob must be true"},
           {R"("shared_expert_intermediate_size": 1024)", R"("shared_expert_intermediate_size": 512)",
            "text config shared_expert_intermediate_size mismatch"},
           {R"("intermediate_size": 6144)", R"("intermediate_size": 4096)", "text config intermediate_size mismatch"},
           {"\n    0,\n    1\n  ],", "\n    0\n  ],", "text config mlp_only_layers mismatch"},
           {"150018\n  ]", "150017\n  ]", "text config eos_token_id mismatch"},
           {R"("rope_theta": 10000.0)", R"("rope_theta": 1000000.0)", "text config rope_theta mismatch"},
           {R"("tie_word_embeddings": false)", R"("tie_word_embeddings": true)",
            "text config tie_word_embeddings must be false"}})
    refuses(flint.with(&SourceModel::config, refused.from, refused.to), refused.error,
            "a Flint config with " + std::string(refused.to) + " was accepted");
  // A flat config is read only for a family whose configs are flat, and
  // Flint's only flat.
  refuses(flint.with(&SourceModel::config, R"("model_type": "flint")", R"("model_type": "qwen3_5_moe_text")"),
          "its config has no text_config", "a flat Qwen config was accepted");
  SourceModel nested = flint;
  nested.config = R"({"model_type":"flint_vl","text_config":)" + flint.config + "}";
  refuses(nested, "no supported model has this architecture (text model type flint)",
          "a Flint config under text_config was accepted");
}

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: model-configuration FIXTURES");
    testFamilies(argv[1]);
    testSources();
    testOneRulePerValue(argv[1]);
    testQuantization(argv[1]);
    testVisionConfig(argv[1]);
    testPrunedExperts(argv[1]);
    testConfigurationCheck(argv[1]);
    testYarn(argv[1]);
    testFlint(argv[1]);
    std::cout << "model configuration: every family, sources, one rule per value, quantization, vision, pruned "
                 "experts, the installer's check, YaRN, Flint PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
