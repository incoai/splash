#pragma once

#include "model/ModelDescriptor.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/GDN.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace splash::test {

// The shapes the families Splash serves run (model::kFamilies), each once, in
// the families' order, which the kernel tests take so that every family's
// kernels are tested, a new family's with no edit here.

// Each family's GDN shape, of the families with GDN layers.
[[nodiscard]] inline std::vector<ops::GdnShape> familyGdnShapes() {
  std::vector<ops::GdnShape> shapes;
  model::forEachFamily([&](const auto &family) {
    if (!family.gdnLayerCount()) return;
    const ops::GdnShape shape{family.gdnKeyHeads,          family.gdnValueHeads,  family.gdnHeadDimension,
                              family.convolutionDimension, family.packedGdnWidth, family.gdnSigmoidGate};
    if (std::find(shapes.begin(), shapes.end(), shape) == shapes.end()) shapes.push_back(shape);
  });
  return shapes;
}

// Each family's full attention of 256-wide heads, the paged and sparse
// attention tests' (q8_flash_attention_metal_test, paged_attention_plan_test):
// its query heads over the KV heads, head dimension and value order of its
// pages. Flint's heads of 128, with its sinks, windows and NoPE, run in
// attention_variants_metal_test against an fp64 reference of their own.
struct AttentionGeometry final {
  uint32_t queryHeads;
  uint32_t kvHeads;
  uint32_t headDimension;
  bool valuesTokenMajor;

  // Its pages of one attention layer in `format`.
  [[nodiscard]] kv::Layout layout(kv::Format format) const {
    return {.attentionLayers = 1, .kvHeads = kvHeads, .headDimension = headDimension, .format = format,
            .valuesTokenMajor = valuesTokenMajor};
  }
  bool operator==(const AttentionGeometry &) const = default;
};
[[nodiscard]] inline std::vector<AttentionGeometry> familyAttentionGeometries() {
  std::vector<AttentionGeometry> geometries;
  model::forEachFamily([&](const auto &family) {
    const kv::Layout layout = family.kvLayout();
    if (layout.headDimension != SPLASH_KV_HEAD_DIMENSION) return;
    const AttentionGeometry geometry{family.attentionQueryHeads, layout.kvHeads, layout.headDimension,
                                     layout.valuesTokenMajor};
    if (std::find(geometries.begin(), geometries.end(), geometry) == geometries.end())
      geometries.push_back(geometry);
  });
  return geometries;
}

} // namespace splash::test
