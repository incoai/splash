// The GLP reader (model/GlpVector.hpp): a synthetic GLP GGUF, as
// captain_vector.py write_gguf writes it, parsed and checked against the
// model's hidden size and layers, and every refusal path. The fixture
// dev/tests/fixtures/glp/synthetic-glp.gguf is a real producer's file (the
// GLP writer's GGUF v3 layout), which the producer's own inspect accepts.

#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestGguf.hpp"
#include "model/GlpVector.hpp"
#include "model/GgufFile.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace gguf = splash::test::gguf;
using splash::model::GlpError;
using splash::model::GlpVector;
using splash::test::rejects;
using splash::test::require;

constexpr uint32_t kHidden = 8, kLayers = 4;

gguf::Tensor f32Tensor(std::string name, std::vector<uint64_t> dims, const std::vector<float> &values) {
  gguf::Bytes data;
  for (const float value : values) gguf::append<float>(data, value);
  return {std::move(name), std::move(dims), splash::model::ggml::kF32, std::move(data)};
}

// A direction of kHidden values, `scale` times a vector whose norm is not 1,
// so that the reader's normalization is visible.
gguf::Tensor direction(std::string name, float scale = 1.0F) {
  std::vector<float> values(kHidden, 0.5F * scale);
  values[0] = scale;
  return f32Tensor(std::move(name), {kHidden}, values);
}

// The metadata of a v1 project file at the post-layer residual stream.
std::vector<gguf::Key> keys() {
  return {gguf::stringKey("glp.mode", "project"),
          gguf::uint32Key("glp.spec_version", 1),
          gguf::stringKey("glp.hook_point", "residual_stream_post_layer"),
          gguf::uint32Key("glp.rank", 1),
          gguf::float32Key("glp.alpha_default", 4.0F),
          gguf::stringKey("glp.base_model", "splash/synthetic"),
          gguf::stringKey("glp.base_revision", "0123456789abcdef"),
          gguf::stringKey("glp.content_sha256", "00")};
}

// `keys` with the value of `name` replaced, or the key removed when `value`
// is null.
std::vector<gguf::Key> with(std::vector<gguf::Key> fileKeys, std::string_view name,
                            const std::optional<gguf::Key> &value) {
  std::erase_if(fileKeys, [&](const gguf::Key &key) { return key.name == name; });
  if (value) fileKeys.push_back(*value);
  return fileKeys;
}

std::filesystem::path write(const std::filesystem::path &directory, const std::string &name,
                            const std::vector<gguf::Key> &fileKeys, const std::vector<gguf::Tensor> &tensors) {
  const std::filesystem::path path = directory / (name + ".gguf");
  splash::test::writeFile(path, gguf::file(fileKeys, tensors));
  return path;
}

// A project file's directions, normalized on load, and its metadata.
void checkProject(const std::filesystem::path &directory) {
  const auto path = write(directory, "project", keys(), {direction("direction.1"), direction("direction.3", 2.0F)});
  const GlpVector vector = GlpVector::load(path, kHidden, kLayers);
  require(vector.alpha == 4.0F && vector.hidden == kHidden && vector.layers == kLayers,
          "a project file's alpha or geometry changed");
  require(vector.directions.size() == kLayers, "a project file's layer count changed");
  require(vector.directions[0].empty() && !vector.directions[1].empty() && vector.directions[2].empty() &&
              !vector.directions[3].empty(),
          "direction.N did not land at layer N");
  require(vector.steeredLayers() == std::vector<uint32_t>{1, 3}, "a project file's steered layers changed");
  for (const uint32_t layer : {1U, 3U}) {
    double squares = 0.0;
    for (const float value : vector.directions[layer]) squares += double{value} * value;
    require(std::abs(std::sqrt(squares) - 1.0) < 1e-6, "a direction is not unit norm after loading");
  }
  require(vector.baseModel == "splash/synthetic" && vector.baseRevision == "0123456789abcdef" &&
              vector.contentSha256 == "00",
          "a project file's provenance changed");
  // The caller's alpha outranks the file's.
  require(GlpVector::load(path, kHidden, kLayers, 0.5F).alpha == 0.5F, "the alpha override did not apply");

  // The v1 defaults: no hook_point is the post-layer stream, no rank is 1, no
  // alpha_default is 1.0.
  const auto defaults = write(directory, "defaults",
                              with(with(with(keys(), "glp.hook_point", std::nullopt), "glp.rank", std::nullopt),
                                   "glp.alpha_default", std::nullopt),
                              {direction("direction.2")});
  const GlpVector plain = GlpVector::load(defaults, kHidden, kLayers);
  require(plain.alpha == 1.0F && plain.steeredLayers() == std::vector<uint32_t>{2},
          "a v1 file's defaults changed");
}

// Every file this build refuses, with the rule its error names.
void checkRefusals(const std::filesystem::path &directory) {
  const auto refuse = [&](const std::string &name, const std::vector<gguf::Key> &fileKeys,
                          const std::vector<gguf::Tensor> &tensors, std::string_view expected) {
    const auto path = write(directory, name, fileKeys, tensors);
    rejects([&] { (void)GlpVector::load(path, kHidden, kLayers); }, expected, name);
  };
  const std::vector<gguf::Tensor> one{direction("direction.1")};

  // The operation travels with the data: absence is add, and an unknown mode
  // is fatal, never a fallback.
  refuse("legacy", {}, one, "no glp.mode");
  refuse("add", with(keys(), "glp.mode", gguf::stringKey("glp.mode", "add")), one, "glp.mode add");
  refuse("unknown-mode", with(keys(), "glp.mode", gguf::stringKey("glp.mode", "banana")), one, "glp.mode 'banana'");

  // The contract version.
  refuse("spec-missing", with(keys(), "glp.spec_version", std::nullopt), one, "glp.spec_version is missing");
  refuse("spec-2", with(keys(), "glp.spec_version", gguf::uint32Key("glp.spec_version", 2)), one,
         "glp.spec_version 2");
  refuse("spec-3", with(keys(), "glp.spec_version", gguf::uint32Key("glp.spec_version", 3)), one,
         "glp.spec_version 3");

  // The apply site.
  refuse("hc-fold", with(keys(), "glp.hook_point", gguf::stringKey("glp.hook_point", "hc_fold_pre_engram")), one,
         "hc_fold_pre_engram");
  refuse("ffn-out", with(keys(), "glp.hook_point", gguf::stringKey("glp.hook_point", "ffn_out_pre_residual")), one,
         "ffn_out_pre_residual");

  // Rank 1 only, and the version-2 keys that would change the alphas.
  refuse("rank-2", with(keys(), "glp.rank", gguf::uint32Key("glp.rank", 2)), one, "glp.rank 2");
  refuse("dir-scales", with(keys(), "glp.dir_scales", gguf::stringKey("glp.dir_scales", "1.0")), one,
         "glp.dir_scales requires spec_version 2");
  refuse("layer-scales", with(keys(), "glp.layer_scales", gguf::stringKey("glp.layer_scales", "1:0.5")), one,
         "glp.layer_scales requires spec_version 2");
  refuse("rank-2-tensor", keys(), {direction("direction.1"), direction("direction.1.1")}, "implements rank 1");

  // The layer mapping and the tensor contract.
  refuse("layer-0", keys(), {direction("direction.0")}, "direction.0 is invalid");
  refuse("past-layers", keys(), {direction("direction.5")}, "past the model's 4 layers");
  // A direction at N == layers has no layer to apply at, and is validated
  // then left unapplied, as llama.cpp's applier does.
  const auto lastLayer = write(directory, "last-layer", keys(), {direction("direction.4")});
  require(GlpVector::load(lastLayer, kHidden, kLayers).steeredLayers().empty(),
          "direction.4 of a 4-layer model was applied");
  refuse("leading-zero", keys(), {direction("direction.01")}, "unexpected tensor 'direction.01'");
  refuse("unexpected-tensor", keys(), {direction("weight")}, "unexpected tensor 'weight'");
  refuse("no-tensors", keys(), {}, "holds no direction.N tensors");
  refuse("f16", keys(), {{"direction.1", {kHidden}, 1, gguf::Bytes(kHidden * 2)}}, "not F32");
  refuse("two-dimensional", keys(), {f32Tensor("direction.1", {kHidden, 1}, std::vector<float>(kHidden, 1.0F))},
         "not 1-D");
  refuse("wrong-hidden", keys(),
         {f32Tensor("direction.1", {kHidden + 1}, std::vector<float>(kHidden + 1, 1.0F))}, "hidden size");
  refuse("zero-norm", keys(), {f32Tensor("direction.1", {kHidden}, std::vector<float>(kHidden, 0.0F))}, "norm 0");
  refuse("non-finite", keys(),
         {f32Tensor("direction.1", {kHidden}, std::vector<float>(kHidden, std::numeric_limits<float>::quiet_NaN()))},
         "non-finite");
  refuse("alpha-non-finite",
         with(keys(), "glp.alpha_default",
              gguf::float32Key("glp.alpha_default", std::numeric_limits<float>::quiet_NaN())),
         one, "glp.alpha_default is not a finite float");

  // A file that is not a GGUF at all.
  const std::filesystem::path text = directory / "text.gguf";
  splash::test::writeFile(text, std::string_view("not a gguf"));
  rejects([&] { (void)GlpVector::load(text, kHidden, kLayers); }, "not a GGUF v3 control vector", "text file");

  // A non-finite override is the caller's error, not the file's.
  const auto path = write(directory, "override", keys(), one);
  rejects([&] { (void)GlpVector::load(path, kHidden, kLayers, std::numeric_limits<float>::quiet_NaN()); },
          "alpha override is not finite", "non-finite override");
}

// The producer's own file: 3 layers of 8 values, alpha 2.5, unit directions,
// for a model of 4 layers (direction.3 is the last applied).
void checkFixture(const std::filesystem::path &path) {
  const GlpVector vector = GlpVector::load(path, 8, 4);
  require(vector.alpha == 2.5F && vector.hidden == 8 && vector.layers == 4, "the fixture's geometry changed");
  require(vector.steeredLayers() == std::vector<uint32_t>{1, 2, 3}, "the fixture's layers changed");
  require(vector.baseModel == "splash/synthetic" && vector.baseRevision == "0000000000000000000000000000000000000000",
          "the fixture's provenance changed");
  for (const uint32_t layer : {1U, 2U, 3U}) {
    double squares = 0.0;
    for (const float value : vector.directions[layer]) squares += double{value} * value;
    require(std::abs(std::sqrt(squares) - 1.0) < 1e-6, "a fixture direction is not unit norm");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: glp-vector FIXTURE.gguf");
    const splash::test::TemporaryDirectory directory("splash-glp-vector");
    checkProject(directory.path());
    checkRefusals(directory.path());
    checkFixture(argv[1]);
    std::cout << "GLP reader tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
