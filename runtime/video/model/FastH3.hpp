#pragma once

// FastVideo's FastH3 students of MiniMax-H3, the variants fasth3-8 (FastVideo-FastH3-8-Step-V2) and fasth3-4
// (FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree, MiniMax-H3 with its LoRA): DMD students with Video Sparse
// Attention on 64-token tiles (video/ops/VsaTable.hpp, video/ops/VsaAttention.hpp) instead of VDN-H3's window softmax and linear
// branch, converted at installation. What sets them apart lives in their DiT's manifest.json:
//   schedule  the trained DMD rungs on the 1000-step clock (8-step V2: 999 874 749 624 500 375 250 125, shift video 10
//             / audio 3; 4-step V1: 999 749 500 250, 12 / 3): sigma = shift r / (1 + (shift - 1) r) of r = rung /
//             1000 in fp32 (FastVideo _set_dmd_schedule), timestep 1 - sigma, one forward per rung, Euler steps
//   vsa       the tile (4 x 4 x 4) and the sparsity (0.8 / 0.9) of the routing
//   adaLN     the block files' tables over the text schedule's distinct timesteps (adaln_timesteps)
// The students were distilled and published for text-to-video (their fastvideo_inference.json: task t2av). A clip's
// first and last frames condition them as FastVideo's pipeline does for any MiniMax-H3 DiT, which the students were
// not trained for: the keyframe rows stay at their noised latents, run at timestep max(t, 0.999) (the fourth adaLN
// slot, tabulated at conversion) and are a dense prefix segment of the VSA tile map.
#include "video/model/Prep.hpp"
#include "video/ops/VsaTable.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace splash::video::model {

// The block layout the FastH3 DiT's manifest.json and block files name: the int8 linears of QUANT_VIDEO_W8
// (metal/abi/QuantFormat.h) as FastH3's blocks hold them.
inline constexpr char kFastH3Format[] = "h3-w8a8-fasth3-v1";

// Whether the DiT in ditDir has FastH3's block layout and manifest contract (pika-6's among them): its manifest.json
// names format kFastH3Format. A directory without manifest.json holds a VDN-H3 DiT (scratch of the installer's
// adaLN tabulation).
[[nodiscard]] bool isFastH3(const std::string &ditDir);
// The directory of a DiT under the weights, by its name (the request's, protocol::validDitName): dit-<name>.
[[nodiscard]] std::string ditDirectory(std::string_view dit);

// How a step moves the latents from the forward's velocity v at sigma to the next rung's sigma'.
//   Euler    x' = x + (sigma' - sigma) v as the rectified-flow Euler step (FastVideo's DMD students)
//   Renoise  Pika H3 Flex's released sampler (sampling.few_step_trajectory): x0 = x + sigma v, then
//            x' = (1 - sigma') x0 + sigma' eps with fresh Gaussian eps per step (video, then audio) from the
//            request's second random stream (seed + 1); the last forward's x0 is the clip
enum class Sampler : int32_t { Euler = 0, Renoise = 1 };

struct FastH3Contract {
  std::vector<int32_t> rungs;  // strictly decreasing, in (0, 1000]
  double videoShift = 0, audioShift = 0;
  ops::VsaSpec vsa;
  std::vector<float> adalnTimesteps;  // the text schedule's distinct timesteps, sorted
  Sampler sampler = Sampler::Euler;   // schedule.sampler: "euler" (default) or "renoise"
  // schedule.sigma_precision "float64": sigmas computed in float64 and rounded to fp32 once (Pika's H3SigmaSchedule),
  // else in fp32 op by op (FastVideo's _set_dmd_schedule)
  bool sigma64 = false;
  uint64_t renoiseSeedOffset = 1;  // schedule.renoise_seed_offset: the fresh noise's stream, seed + offset
};
// The contract of the FastH3 DiT in ditDir (its manifest.json).
[[nodiscard]] FastH3Contract fastH3Contract(const std::string &ditDir);

} // namespace splash::video::model
