#pragma once

// A GLP control vector (the GLP spec,
// https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md): per-layer
// directions applied to the post-layer residual stream as a projection,
//
//   h <- h - alpha * (h . d_hat) * d_hat
//
// where d_hat is the layer's unit-norm direction. The file is a GGUF v3
// container whose `glp.*` metadata says which operation the data is for; the
// operation travels with the data because an additive consumer loading a
// projective direction applies cleanly and is silently wrong.
//
// This build implements one contract: glp.spec_version 1, glp.mode project,
// glp.hook_point residual_stream_post_layer, glp.rank 1. Everything else is
// refused with a named error rather than approximated. A file with no glp.*
// metadata is a legacy llama.cpp additive control vector, which this build has
// no path for, so it is refused too.
//
// A project file is never merged with another control vector: two projections
// do not compose into a projection along the sum of their directions, so a
// caller names one file.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::model {

// A GLP file this build cannot apply, with the rule it broke.
class GlpError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// The GLP contract this build implements, as the file states it.
struct GlpVector final {
  // The projection strength: glp.alpha_default, or the caller's override.
  // Never folded into the directions: projection is quadratic in the
  // direction's norm, so a strength baked into the data would not mean what a
  // caller expects.
  float alpha = 1.0F;
  // The model's hidden size and layer count the file was checked against.
  uint32_t hidden = 0;
  uint32_t layers = 0;
  // directions[layer] is the unit-norm direction of layer `layer`, `hidden`
  // values, empty for a layer the file does not steer. direction.N applies at
  // layer N, with no offset.
  std::vector<std::vector<float>> directions;
  // Provenance, informational: glp.base_model, glp.base_revision and
  // glp.content_sha256, empty when the file omits them.
  std::string baseModel;
  std::string baseRevision;
  std::string contentSha256;

  // Reads `path` and validates it against the model's `hidden` and `layers`
  // (the loaded target's), with `alpha` overriding glp.alpha_default when
  // given. Throws GlpError for a file this build cannot apply, naming the
  // rule and the value that broke it.
  [[nodiscard]] static GlpVector load(const std::filesystem::path &path, uint32_t hidden, uint32_t layers,
                                      std::optional<float> alpha = std::nullopt);

  // The layers the file steers, ascending.
  [[nodiscard]] std::vector<uint32_t> steeredLayers() const;
};

} // namespace splash::model
