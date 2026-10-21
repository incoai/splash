#include "model/ControlVector.hpp"

#include "model/GgufFile.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace splash::model {
namespace {

constexpr std::string_view kArchitecture = "controlvector";
constexpr std::string_view kDirection = "direction.";

// The layer a direction.L tensor names; none for another name.
std::optional<uint32_t> directionLayer(std::string_view name) {
  if (!name.starts_with(kDirection)) return std::nullopt;
  const std::string_view digits = name.substr(kDirection.size());
  uint32_t layer = 0;
  const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
  if (digits.empty() || parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size())
    return std::nullopt;
  return layer;
}

} // namespace

std::string_view controlVectorModeName(ControlVectorMode mode) noexcept {
  return mode == ControlVectorMode::Project ? "project" : "add";
}

ControlVectorTable readControlVector(const ControlVectorOptions &options, std::string_view architecture,
                                     uint32_t layers, uint32_t hidden) {
  const std::string file = options.path.string();
  const auto refuse = [&](const std::string &why) { return GgufError("control vector " + file + ": " + why); };
  if (!options || !layers || !hidden) throw std::invalid_argument("a control vector needs a file and a target");
  if (!std::isfinite(options.scale)) throw refuse("the scale is not finite");
  if (options.firstLayer > options.lastLayer || options.firstLayer >= layers)
    throw refuse("the layer range " + std::to_string(options.firstLayer) + "-" + std::to_string(options.lastLayer) +
                 " lies outside the target's " + std::to_string(layers) + " layers");
  const OpenGguf gguf({options.path});
  const GgufFile &header = gguf.file();
  if (header.metadata().architecture() != kArchitecture)
    throw refuse("general.architecture is not " + std::string(kArchitecture));
  if (const std::optional<std::string> hint = header.metadata().stringValue("controlvector.model_hint");
      hint && !hint->empty() && *hint != architecture)
    throw refuse("it is for " + *hint + ", the target is " + std::string(architecture));

  ControlVectorTable table;
  table.mode = options.mode;
  table.scale = options.scale;
  table.hidden = hidden;
  table.directions.assign(uint64_t{layers} * hidden, 0.0F);
  std::vector<bool> present(layers, false);
  for (const GgufTensor &tensor : header.tensors()) {
    const std::optional<uint32_t> layer = directionLayer(tensor.name);
    if (!layer) throw refuse("tensor " + tensor.name + " is not a direction.L");
    if (!*layer || *layer >= layers)
      throw refuse(tensor.name + " names no layer of the target's 1 to " + std::to_string(layers - 1));
    const std::span<float> row(table.directions.data() + uint64_t{*layer} * hidden, hidden);
    if (tensor.type != ggml::kF32 || tensor.dims.size() != 1 || tensor.dims[0] != hidden ||
        tensor.bytes != row.size_bytes())
      throw refuse(tensor.name + " is not F32 of " + std::to_string(hidden) + " values");
    if (present[*layer]) throw refuse(tensor.name + " appears twice");
    present[*layer] = true;
    header.source(tensor.file).readData(tensor.offset, {reinterpret_cast<uint8_t *>(row.data()), row.size_bytes()});
    double squares = 0.0;
    for (const float value : row) {
      if (!std::isfinite(value)) throw refuse(tensor.name + " holds a value that is not finite");
      squares += double{value} * value;
    }
    const bool steered = squares > 0.0 && *layer >= options.firstLayer && *layer <= options.lastLayer;
    if (!steered) {
      std::fill(row.begin(), row.end(), 0.0F);
      continue;
    }
    // A projection removes the component along the unit direction.
    if (options.mode == ControlVectorMode::Project) {
      const double norm = std::sqrt(squares);
      for (float &value : row) value = static_cast<float>(value / norm);
    }
  }
  gguf.checkUnchanged();
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const float *row = table.directions.data() + uint64_t{layer} * hidden;
    if (std::any_of(row, row + hidden, [](float value) { return value != 0.0F; })) table.layers.push_back(layer);
  }
  if (table.layers.empty()) throw refuse("it steers no layer in the range");
  // What it does, which no other vector does: its mode, scale and the
  // directions of the layers it steers. The scale takes the digits that read
  // back as the same float.
  std::ostringstream canonical;
  canonical << "splash-control-vector-v1 " << controlVectorModeName(table.mode) << ' '
            << std::setprecision(std::numeric_limits<float>::max_digits10) << table.scale << " hidden " << hidden;
  for (const uint32_t layer : table.layers) canonical << ' ' << layer;
  canonical << '\n';
  std::string text = canonical.str();
  for (const uint32_t layer : table.layers) {
    const char *bytes = reinterpret_cast<const char *>(table.directions.data() + uint64_t{layer} * hidden);
    text.append(bytes, uint64_t{hidden} * sizeof(float));
  }
  table.sha256 = weightDigest(std::string_view(text));
  return table;
}

bool ControlVector::steers(uint32_t layer) const noexcept {
  return std::binary_search(layers.begin(), layers.end(), layer);
}

std::string ControlVector::describe() const {
  std::ostringstream text;
  text << controlVectorModeName(mode) << ", layers " << layers.front() << ".." << layers.back() << " ("
       << layers.size() << " steered), scale " << scale;
  return text.str();
}

ControlVector loadControlVector(metal::MetalBackend &backend, const ControlVectorTable &table) {
  if (table.layers.empty() || table.directions.empty()) throw std::invalid_argument("an empty control vector");
  ControlVector result;
  result.mode = table.mode;
  result.scale = table.scale;
  result.layers = table.layers;
  result.sha256 = table.sha256;
  const uint64_t bytes = table.directions.size() * sizeof(float);
  result.directions = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "control-vector");
  std::memcpy(result.directions.contents(), table.directions.data(), bytes);
  return result;
}

} // namespace splash::model
