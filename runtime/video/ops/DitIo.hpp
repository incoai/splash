#pragma once

// DiT input embedding and final layer (kernels/dit/io.metal), once per forward around the 50 blocks.

#include "video/metal/Encoder.hpp"
#include "video/ops/Geometry.hpp"

#include <string>
#include <vector>

namespace splash::video::ops {

struct IoBuffers {
  BufferView residual, tags, text, adaFinal;
  BufferView videoIn, audioIn, videoOut, audioOut;
};

// X = [refined text | video_patch_proj(conditioning) | audio_patch_proj(audio) | video_patch_proj(video)], bf16;
// videoIn holds the conditioning rows, then the target rows. Projections fp32 [K][5376] (transposed) + bias [5376].
void embed(Encoder &e, const SequenceLayout &layout, BufferView videoProj, BufferView videoProjBias,
           BufferView audioProj, BufferView audioProjBias, const IoBuffers &x);
// final_layer on the target audio and video rows -> fp32 velocities: bf16 norm [5376], fp32 outputs [K][5376] + [K].
void finalLayer(Encoder &e, const SequenceLayout &layout, BufferView norm, BufferView videoOut, BufferView videoOutBias,
                BufferView audioOut, BufferView audioOutBias, const IoBuffers &x);

} // namespace splash::video::ops
