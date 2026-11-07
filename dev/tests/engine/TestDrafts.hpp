#pragma once

#include "model/DFlashDraft.hpp"
#include "model/Flint.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen4Exp.hpp"
#include "ops/Vision.hpp"

#include <array>
#include <cstdint>

namespace splash::test {

// The DFlash2 drafts the configs in dev/tests/fixtures/model-configs
// describe (model/ModelDescriptor.mm readDraftLayout): the released drafts
// of Qwen3.8-27B, Qwen3.6-35B-A3B and Qwen3.5-122B-A10B (Ornith-1.5-122B's,
// trained to fill a block of 16), one of made-up sizes for
// Qwen3.8-Flash-Next, and Flint's test draft of one layer of random weights,
// whose mask token is Flint's first reserved token.
[[nodiscard]] inline model::DFlashDraftLayout qwen3_8Draft() {
  constexpr model::Qwen3_8Layout target;
  constexpr std::array<uint32_t, 5> targetLayers{5, 19, 33, 47, 61};
  return model::draftLayout(target.hiddenSize, target.vocabularySize, 5, 17408, 10'000'000.0F, targetLayers);
}
[[nodiscard]] inline model::DFlashDraftLayout qwen3_6MoeDraft() {
  constexpr model::Qwen3_6MoeLayout target;
  constexpr std::array<uint32_t, 8> targetLayers{1, 6, 11, 16, 22, 27, 32, 37};
  return model::draftLayout(target.hiddenSize, target.vocabularySize, 6, 6144, 10'000'000.0F, targetLayers);
}
[[nodiscard]] inline model::DFlashDraftLayout qwen3_5_122BDraft() {
  constexpr model::Qwen3_6MoeLayout target = model::kQwen3_5_122BLayout;
  constexpr std::array<uint32_t, 8> targetLayers{1, 7, 14, 20, 26, 32, 39, 45};
  return model::draftLayout(target.hiddenSize, target.vocabularySize, 6, 9216, 10'000'000.0F, targetLayers);
}
[[nodiscard]] inline model::DFlashDraftLayout flashNextDraft() {
  constexpr model::Qwen4ExpLayout target;
  constexpr std::array<uint32_t, 4> targetLayers{3, 17, 31, 47};
  return model::draftLayout(target.hiddenSize, target.vocabularySize, 3, 5120, 1'000'000.0F, targetLayers);
}

[[nodiscard]] inline model::DFlashDraftLayout flintDraft() {
  constexpr model::FlintLayout target;
  constexpr std::array<uint32_t, 5> targetLayers{1, 12, 24, 35, 47};
  return model::draftLayout(target.hiddenSize, target.vocabularySize, 1, 6144, 10'000'000.0F, targetLayers);
}

// The vision tower of Qwen3.8-27B, projecting into its hidden width.
[[nodiscard]] constexpr ops::VisionLayout qwen3_8Vision() {
  return {.outputHiddenSize = model::Qwen3_8Layout{}.hiddenSize};
}

} // namespace splash::test
