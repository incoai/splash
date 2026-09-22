#pragma once

#include "FlashDenseTraversal.hpp"
#include <stdexcept>
#include <string_view>

namespace splash::flash {

inline constexpr const char *kFlashPrefillDenseTilesSemantics =
    ";dense-bf16-cached-r2048-inspected-shapes-m128n64-simd4or8-whole-k-v1";

[[nodiscard]] inline bool parseFlashPrefillDenseTilesFlag(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_PREFILL_DENSE_TILES must be 0 or 1");
}
[[nodiscard]] bool flashPrefillDenseTilesEnabled();

struct FlashPrefillDenseTilePlan final {
  uint32_t tileRows = 0, tileOutputs = 0, simdGroups = 0;
  FlashDenseTraversal traversal = FlashDenseTraversal::ColumnFast;
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return tileRows != 0; }
};

// Actual2048-row model input/output fixtures were byte-exact for these whole-K
// variants. Selection is restricted by the caller to cached affine projections;
// original BF16 router matrices retain the public whole-K route's normal tiles.
[[nodiscard]] constexpr FlashPrefillDenseTilePlan flashPrefillDenseTilePolicy(
    uint32_t rows, uint32_t outputs, uint32_t inputs) noexcept {
  if (rows != 2048) return {};
  if (inputs == 10240 && outputs == 320)
    return {128,64,8,FlashDenseTraversal::ColumnFast};
  if ((inputs == 2560 && (outputs == 10240 || outputs == 6144)) ||
      (inputs == 6144 && outputs == 2560))
    return {128,64,4,FlashDenseTraversal::Swizzle4};
  if (inputs == 2560 && outputs == 12288)
    return {128,64,4,FlashDenseTraversal::Swizzle8};
  if (inputs == 2560 && outputs == 512)
    return {128,64,8,FlashDenseTraversal::Swizzle2};
  if (inputs == 2560 && outputs == 640)
    return {128,64,8,FlashDenseTraversal::Swizzle8};
  return {};
}

// Production scope is the validated 48-layer Flash architecture and its known
// projection role/shape pairs. Primitive captures cover representative HCdown,
// GDN and QSA roles. Layer 1 PLEkey was qualified by matched full-model hidden/
// logits and service output parity, rather than by the ten-role primitive capture.
// MTP, shared gate/up, unknown roles and malformed layer prefixes fall back.
[[nodiscard]] constexpr FlashPrefillDenseTilePlan flashPrefillDenseTilePolicy(
    std::string_view prefix,uint32_t rows,uint32_t outputs,uint32_t inputs) noexcept {
  if (rows != 2048) return {};
  if (prefix == "language_model.model.hyper_connection_mixer.input_mix_weight_down")
    return inputs == 10240 && outputs == 320 ? flashPrefillDenseTilePolicy(rows,outputs,inputs)
                                           : FlashPrefillDenseTilePlan{};
  constexpr std::string_view leading = "language_model.model.layers.";
  if (!prefix.starts_with(leading)) return {};
  const auto tail = prefix.substr(leading.size());
  const auto split = tail.find('.');
  if (split == std::string_view::npos || !split || split > 2 ||
      (split > 1 && tail.front() == '0')) return {};
  uint32_t layer = 0;
  for (char digit : tail.substr(0,split)) {
    if (digit < '0' || digit > '9') return {};
    layer = layer*10+uint32_t(digit-'0');
  }
  if (layer >= 48) return {};
  const auto role = tail.substr(split+1);
  bool qualified = false;
  if (role == "attn_hyper_connection.input_mix_weight_down" ||
      role == "mlp_hyper_connection.input_mix_weight_down")
    qualified = inputs == 10240 && outputs == 320;
  else if (layer == 1 && role == "ple.key_proj")
    qualified = inputs == 2560 && outputs == 10240;
  else if (layer%4 != 3) {
    if (role == "linear_attn.in_proj_qkv") qualified = inputs == 2560 && outputs == 10240;
    else if (role == "linear_attn.in_proj_z") qualified = inputs == 2560 && outputs == 6144;
    else if (role == "linear_attn.out_proj") qualified = inputs == 6144 && outputs == 2560;
  } else {
    if (role == "self_attn.q_proj") qualified = inputs == 2560 && outputs == 12288;
    else if (role == "self_attn.k_proj" || role == "self_attn.v_proj")
      qualified = inputs == 2560 && outputs == 512;
    else if (role == "self_attn.indexer.index_qk_proj") qualified = inputs == 2560 && outputs == 640;
    else if (role == "self_attn.o_proj") qualified = inputs == 6144 && outputs == 2560;
  }
  return qualified ? flashPrefillDenseTilePolicy(rows,outputs,inputs) : FlashPrefillDenseTilePlan{};
}

} // namespace splash::flash
