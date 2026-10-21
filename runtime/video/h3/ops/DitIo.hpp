#pragma once

// DiT input embedding and final layer (kernels/dit/io.metal), once per forward around the 50 blocks.

#include "video/h3/engine/Encoder.hpp"
#include "video/h3/model/DitWeights.hpp"
#include "video/h3/model/Geometry.hpp"

#include <string>
#include <vector>

namespace h3::ops {

struct IoBuffers {
  BufferView residual, tags, text, adaFinal;
  BufferView videoIn, audioIn, videoOut, audioOut;
};

// X = [refined text | audio_patch_proj(audio) | video_patch_proj(video)], bf16.
void embed(Encoder &e, const model::SequenceLayout &layout, const model::DitIoWeights &w, const IoBuffers &x);
// final_layer on the audio and video rows -> fp32 velocities.
void finalLayer(Encoder &e, const model::SequenceLayout &layout, const model::DitIoWeights &w, const IoBuffers &x);

} // namespace h3::ops
