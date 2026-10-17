#pragma once

#include "model/DFlashDraft.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "ops/Vision.hpp"

#include <array>
#include <cstdint>

namespace splash::test {

// The DFlash2 drafts the configs in dev/tests/fixtures/model-configs
// describe (model/ModelDescriptor.mm readDraftLayout): the released drafts
// of Qwen3.8-27B and Qwen3.6-35B-A3B.
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

// The vision tower of Qwen3.8-27B, projecting into its hidden width.
[[nodiscard]] constexpr ops::VisionLayout qwen3_8Vision() {
  return {.outputHiddenSize = model::Qwen3_8Layout{}.hiddenSize};
}

} // namespace splash::test
