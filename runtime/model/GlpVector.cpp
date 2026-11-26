#include "model/GlpVector.hpp"

#include "model/GgufFile.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

namespace splash::model {
namespace {

constexpr std::string_view kSpecUrl = "https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md";

// A file this build cannot apply, with the rule it broke and where the rule
// is written.
[[noreturn]] void refuse(const std::string &message) {
  throw GlpError(message + " (GLP spec: " + std::string(kSpecUrl) + ")");
}

struct DirectionName final {
  uint32_t layer = 0;
  uint32_t index = 0;
};

// direction.N -> (N, 0); direction.N.j -> (N, j). A leading zero is refused so
// that direction.01 cannot alias direction.1, and anything else is no
// direction name at all.
std::optional<DirectionName> parseDirectionName(std::string_view name) {
  constexpr std::string_view prefix = "direction.";
  if (!name.starts_with(prefix)) return std::nullopt;
  const std::string_view rest = name.substr(prefix.size());
  const size_t dot = rest.find('.');
  const std::string_view layerText = rest.substr(0, dot);
  const std::string_view indexText = dot == std::string_view::npos ? std::string_view{} : rest.substr(dot + 1);
  const auto number = [](std::string_view text, uint32_t &value) {
    if (text.empty() || (text.size() > 1 && text.front() == '0')) return false;
    uint32_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return false;
    value = parsed;
    return true;
  };
  DirectionName parsed;
  if (!number(layerText, parsed.layer)) return std::nullopt;
  if (dot != std::string_view::npos && !number(indexText, parsed.index)) return std::nullopt;
  return parsed;
}

// The file's alpha_default, or the caller's override, as a finite float.
float readAlpha(const GgufMetadata &metadata, std::optional<float> override) {
  float alpha = 1.0F;
  if (const std::optional<double> value = metadata.floatValue("glp.alpha_default")) {
    if (!std::isfinite(*value) || std::abs(*value) > std::numeric_limits<float>::max())
      refuse("glp.alpha_default is not a finite float");
    alpha = static_cast<float>(*value);
  }
  if (override) {
    if (!std::isfinite(*override)) throw std::invalid_argument("GLP alpha override is not finite");
    alpha = *override;
  }
  return alpha;
}

// The unit-norm direction of layer `layer`'s tensor, read through its file's
// source.
std::vector<float> readDirection(const GgufFile &file, const GgufTensor &tensor, uint32_t layer, uint32_t hidden) {
  std::vector<float> values(hidden);
  file.source(tensor.file).readData(
      tensor.offset,
      std::span<uint8_t>(reinterpret_cast<uint8_t *>(values.data()), uint64_t{hidden} * sizeof(float)));
  double squares = 0.0;
  for (const float value : values) {
    if (!std::isfinite(value)) refuse("direction." + std::to_string(layer) + " holds a non-finite value");
    squares += double{value} * value;
  }
  const double norm = std::sqrt(squares);
  if (!(norm > 0.0)) refuse("direction." + std::to_string(layer) + " has norm 0");
  const float scale = static_cast<float>(1.0 / norm);
  for (float &value : values) value *= scale;
  return values;
}

} // namespace

GlpVector GlpVector::load(const std::filesystem::path &path, uint32_t hidden, uint32_t layers,
                          std::optional<float> alpha) {
  if (!hidden || !layers) throw std::invalid_argument("a GLP vector needs the model's hidden size and layers");
  try {
    const OpenGguf file({path});
    const GgufMetadata &metadata = file.file().metadata();

    // 1. The operation travels with the data. Absence means add, which this
    // build has no path for; an unknown value is fatal, never a fallback.
    const std::optional<std::string> mode = metadata.stringValue("glp.mode");
    if (!mode)
      refuse("no glp.mode: a legacy llama.cpp additive control vector, which this build does not apply; "
             "re-export it as glp.mode=project, spec_version 1");
    if (*mode == "add")
      refuse("glp.mode add (additive) is not implemented: this build applies project mode only "
             "(h <- h - alpha (h . d) d)");
    if (*mode != "project") refuse("glp.mode '" + *mode + "' not supported: this build implements project only");

    // 2. The contract version, which says how to read the rest.
    const std::optional<uint64_t> spec = metadata.unsignedValue("glp.spec_version");
    if (!spec) refuse("glp.spec_version is missing: this build implements spec_version 1");
    if (*spec != 1)
      refuse("glp.spec_version " + std::to_string(*spec) +
             " not supported: this build implements v1 (project, residual_stream_post_layer, rank 1)");

    // 3. The apply site. A vector is calibrated for one site; applying it at
    // another degrades silently, so a mismatch is fatal. Absence is the v1
    // site, the only one this build applies at.
    const std::string hook = metadata.stringValue("glp.hook_point").value_or("residual_stream_post_layer");
    if (hook != "residual_stream_post_layer") {
      if (hook == "hc_fold_pre_engram")
        refuse("glp.hook_point hc_fold_pre_engram not supported: it needs spec_version 3, the mean over "
               "hyper-connection copies and the companion attention writer; this build applies "
               "residual_stream_post_layer only");
      refuse("glp.hook_point '" + hook + "' not supported: this build applies residual_stream_post_layer only");
    }

    // 4. Rank 1 only. A rank-k file must be applied as a subspace, never as
    // its first direction alone.
    if (const std::optional<uint64_t> rank = metadata.unsignedValue("glp.rank"); rank && *rank != 1)
      refuse("glp.rank " + std::to_string(*rank) +
             " not supported: this build implements rank 1 (one direction per layer)");
    for (const char *key : {"glp.dir_scales", "glp.layer_scales"})
      if (metadata.stringValue(key)) refuse(std::string(key) + " requires spec_version 2: this build implements v1");

    const float strength = readAlpha(metadata, alpha);

    // 5. The directions: direction.N at layer N, F32, 1-D, the model's hidden
    // size, uniform across tensors.
    const std::vector<GgufTensor> &tensors = file.file().tensors();
    if (tensors.empty()) refuse("holds no direction.N tensors");
    std::vector<std::vector<float>> directions(layers);
    for (const GgufTensor &tensor : tensors) {
      const std::optional<DirectionName> name = parseDirectionName(tensor.name);
      if (!name) refuse("unexpected tensor '" + tensor.name + "': a GLP file holds direction.N tensors only");
      const std::string layer = std::to_string(name->layer);
      if (name->index != 0)
        refuse("tensor '" + tensor.name + "' is direction " + std::to_string(name->index) + " of layer " + layer +
               ": this build implements rank 1");
      if (name->layer == 0) refuse("direction.0 is invalid: direction.N applies at layer N, N >= 1");
      if (name->layer > layers)
        refuse("direction." + layer + " applies past the model's " + std::to_string(layers) + " layers");
      if (tensor.type != ggml::kF32)
        refuse("direction." + layer + " is " + ggmlTypeName(tensor.type) + ", not F32");
      if (tensor.dims.size() != 1) refuse("direction." + layer + " is not 1-D");
      if (tensor.dims[0] != hidden)
        refuse("direction." + layer + " holds " + std::to_string(tensor.dims[0]) +
               " values, the model's hidden size is " + std::to_string(hidden));
      // A direction at N == layers has no layer to apply at (layers are
      // 0..layers-1), which llama.cpp's applier never reads either: validate
      // it, then leave it unapplied. A file's direction count may therefore
      // equal the model's layer count.
      if (name->layer == layers) {
        (void)readDirection(file.file(), tensor, name->layer, hidden);
        continue;
      }
      if (!directions[name->layer].empty()) refuse("direction." + layer + " appears twice");
      directions[name->layer] = readDirection(file.file(), tensor, name->layer, hidden);
    }

    GlpVector vector;
    vector.alpha = strength;
    vector.hidden = hidden;
    vector.layers = layers;
    vector.directions = std::move(directions);
    vector.baseModel = metadata.stringValue("glp.base_model").value_or("");
    vector.baseRevision = metadata.stringValue("glp.base_revision").value_or("");
    vector.contentSha256 = metadata.stringValue("glp.content_sha256").value_or("");
    return vector;
  } catch (const GgufError &error) {
    throw GlpError(std::string("not a GGUF v3 control vector: ") + error.what());
  }
}

std::vector<uint32_t> GlpVector::steeredLayers() const {
  std::vector<uint32_t> layers;
  for (uint32_t layer = 0; layer < directions.size(); ++layer)
    if (!directions[layer].empty()) layers.push_back(layer);
  return layers;
}

} // namespace splash::model
