#pragma once

// RFC 192 scaffolding: fused prompt-causal attention.
// See docs/rfc/192-fused-attn.md and
// https://github.com/incoai/splash/issues/192.

// https://github.com/incoai/splash/issues/192.

#include <cstdint>

namespace splash::ops::rfc192 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_FUSED_PROMPT_ATTN";

// Fused-scale availability: these (kvHeads, group) combos ship a fused
// scale-in-softmax split kernel in prefill/attention_q8.metal
// (ScaleInSoftmax=true), so the scale+causal-select epilogue bonsai fused
// is already fused here — the follow-up only pre-builds them.
[[nodiscard]] constexpr bool fusedScaleAvailable(std::uint32_t kvHeads,
                                                std::uint32_t group) noexcept {
  return (kvHeads == 4 && group == 6) || (kvHeads == 2 && group == 8);
}

// Warm-up roster: split + reduce pairs must both be resident before the
// first prefill or the reduce stalls behind the split's JIT. BF16 shares
// the page loop with fused scale, so it joins the same roster.
inline constexpr const char* kWarmPrefillPipelines[] = {
    "prefill_attention_q8_split",
    "prefill_attention_q8_reduce",
    "prefill_attention_q8_split_kv2_g8",
    "prefill_attention_q8_reduce_kv2_g8",
    "prefill_attention_bf16_split",
    "prefill_attention_bf16_split_kv2_g8",
};
inline constexpr std::uint32_t kWarmPrefillPipelineCount = 6;

// NOTE (deferred, with reason): fusing split+reduce into one dispatch is
// NOT proposed here — split grids over (KVHeads, tiles, splits) while
// reduce grids over (KVHeads, M, tiles), so one persistent grid cannot
// cover both without padding either phase. The measured bonsai win came
// from the scale epilogue fusion, which already ships above.

static_assert(fusedScaleAvailable(4, 6) && fusedScaleAvailable(2, 8));
static_assert(!fusedScaleAvailable(4, 8) && !fusedScaleAvailable(8, 8));
static_assert(kWarmPrefillPipelineCount == 6);

}  // namespace splash::ops::rfc192
