#pragma once

#include "model/ModelDescriptor.hpp"
#include "model/SafetensorsImage.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/GDN.hpp"
#include "ops/KernelCalibration.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace splash::test {

// The shapes the families Splash serves run (model::kFamilies), each once, in
// the families' order, which the kernel tests take so that every family's
// kernels are tested, a new family's with no edit here.

// A family's GDN shape.
template <class Family> [[nodiscard]] ops::GdnShape gdnShape(const Family &family) {
  return {family.gdnKeyHeads,          family.gdnValueHeads,  family.gdnHeadDimension,
          family.convolutionDimension, family.packedGdnWidth, family.gdnSigmoidGate};
}

// Each family's GDN shape, of the families with GDN layers.
[[nodiscard]] inline std::vector<ops::GdnShape> familyGdnShapes() {
  std::vector<ops::GdnShape> shapes;
  model::forEachFamily([&](const auto &family) {
    if (!family.gdnLayerCount()) return;
    const ops::GdnShape shape = gdnShape(family);
    if (std::find(shapes.begin(), shapes.end(), shape) == shapes.end()) shapes.push_back(shape);
  });
  return shapes;
}

// A family's full attention: its query heads over the KV heads and head
// dimension of its pages, and how it reads them.
struct AttentionGeometry final {
  uint32_t queryHeads;
  uint32_t kvHeads;
  uint32_t headDimension;
  kv::Reads reads;

  // Its pages of one attention layer in `format`, in its reader's value order.
  [[nodiscard]] kv::Layout layout(kv::Format format) const {
    return {.attentionLayers = 1, .kvHeads = kvHeads, .headDimension = headDimension, .format = format,
            .valuesTokenMajor = kv::valuesTokenMajorFor(reads), .reads = reads};
  }
  bool operator==(const AttentionGeometry &) const = default;
};
// The Apple10 attention row (KernelCalibration.hpp) whose dense splits score
// INT8 queries over INT8 pages from int8QueriesFromPages pages a split reads,
// for tests that choose the queries by the row, with int8 activations on:
// kNever keeps bf16 queries, 1 takes INT8 queries in every split. A plan with
// int8 activations off scores bf16 queries whatever its row.
[[nodiscard]] constexpr ops::AttentionCalibration attentionRow(uint32_t int8QueriesFromPages) noexcept {
  ops::AttentionCalibration row = ops::kApple10Attention;
  row.int8QueriesFromPages = int8QueriesFromPages;
  return row;
}

// The history splits of each tile of a sequence of `rows` rows after
// `history` tokens (ops::PagedAttention's rule, written out): one tile, a
// split per 16 pages it reads, between 32 and 128; more tiles, a whole split
// per `pagesPerSplit` pages they read, at least 32 shared out over them and
// at most 32, nor 256 over them; with a window, the pages a window reads at
// most, a split each. A sparse sequence of more than one tile reads its
// lists, the shared splits alone.
[[nodiscard]] constexpr uint32_t attentionSplits(uint32_t rows, uint32_t history, bool sparse,
                                                 uint32_t windowTokens = 0, uint32_t pagesPerSplit = 32) noexcept {
  const uint32_t tiles = (rows + 7) / 8;
  const uint32_t windowPages = (windowTokens + 7 + 31) / 32 + 1;
  uint64_t pages = sparse && tiles > 1 ? 0 : (uint64_t{history} + rows + 31) / 32;
  if (windowTokens) pages = std::min<uint64_t>(pages, windowPages);
  const uint32_t fewest = std::clamp(32U / tiles, 1U, 32U);
  const uint32_t most = std::clamp(256U / tiles, fewest, 32U);
  const auto splits = static_cast<uint32_t>(tiles == 1 ? std::clamp<uint64_t>((pages + 15) / 16, 32, 128)
                                                       : std::clamp<uint64_t>(pages / pagesPerSplit, fewest, most));
  return windowTokens ? std::min(splits, windowPages) : splits;
}

// A family's attention geometry.
template <class Family> [[nodiscard]] AttentionGeometry attentionGeometry(const Family &family) {
  const kv::Layout layout = family.kvLayout();
  return {family.attentionQueryHeads, layout.kvHeads, layout.headDimension, layout.reads};
}
// Each family's full attention of 256-wide heads, the paged and sparse
// attention tests' (q8_flash_attention_metal_test, paged_attention_plan_test).
// Flint's heads of 128, with its sinks, windows and NoPE, run in
// attention_variants_metal_test (its kGeometry) against an fp64 reference of
// their own.
[[nodiscard]] inline std::vector<AttentionGeometry> familyAttentionGeometries() {
  std::vector<AttentionGeometry> geometries;
  model::forEachFamily([&](const auto &family) {
    const AttentionGeometry geometry = attentionGeometry(family);
    if (geometry.headDimension != SPLASH_KV_HEAD_DIMENSION) return;
    if (std::find(geometries.begin(), geometries.end(), geometry) == geometries.end())
      geometries.push_back(geometry);
  });
  return geometries;
}

// The norms the families that `runs` takes store, as ops::NormWeights::float32
// values: bf16 first where one of them loads an MLX checkpoint whose images
// keep its bf16 norms under some naming (safetensors::float32Norms), then F32,
// as every family's GGUFs store them. The kernels of a shape that only
// families of F32 norms run (Qwen3.8-Flash-Next's, and Flint's, whose MLX
// images widen its norms to F32) are compiled for F32 norms alone.
template <class Runs> [[nodiscard]] std::vector<bool> normsOfFamilies(Runs runs) {
  using model::safetensors::ModuleNames;
  bool bfloat16 = false;
  model::forEachFamily([&](const auto &family) {
    if (!runs(family)) return;
    for (const ModuleNames names : {ModuleNames::Mlx, ModuleNames::MlxLm, ModuleNames::Transformers})
      bfloat16 = bfloat16 || !model::safetensors::float32Norms(names, family.ggufArchitecture);
  });
  return bfloat16 ? std::vector<bool>{false, true} : std::vector<bool>{true};
}
// The norms the families of a GDN shape store.
[[nodiscard]] inline std::vector<bool> familyNorms(const ops::GdnShape &shape) {
  return normsOfFamilies([&](const auto &family) { return family.gdnLayerCount() && gdnShape(family) == shape; });
}
// The norms the families of an attention geometry store.
[[nodiscard]] inline std::vector<bool> familyNorms(const AttentionGeometry &geometry) {
  return normsOfFamilies([&](const auto &family) { return attentionGeometry(family) == geometry; });
}

} // namespace splash::test
