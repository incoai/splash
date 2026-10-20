#pragma once

// The engine process of a video model: `serve-native MODEL_DIRECTORY` of an
// assembly whose record names an h3-video model. It speaks the native
// protocol on stdin and stdout as a language model's engine does, announcing
// FeatureVideo, and takes VideoRequests one at a time, and Cancels; every
// other client frame ends the stream. A request runs every phase on the GPU
// in this process, in order: the text encoder (TextEncoder.hpp, loaded for
// the request), then the vendored MiniMax-H3 engine through its C ABI
// (h3/h3.h): request preparation, the DiT, streamed from SSD, the TAEH3 preview
// when asked, the video VAE, the audio VAE when asked and the mp4 writer.
// Each phase reports VideoProgress, the denoising each step; the preview and
// the clip arrive as VideoOutputs. A Cancel stops the request at the next
// step or phase boundary. The engine's own log lines go to stderr.
//
// Two options serve the identity tests (DEVELOPMENT.md, Video):
//   --hidden-states NPZ  every request is conditioned on the f32 [L][5120]
//                        `hidden` of NPZ instead of its encoded prompt;
//   --keep DIRECTORY     each request writes its prompt's hidden states
//                        (hidden.npz), final latents (final.npz) and video
//                        frames (frames.rgb, uint8 RGB) into
//                        DIRECTORY/REQUEST_ID.

#include <filesystem>

namespace splash::video {

// Whether the assembly at modelRoot holds a video model: its model.json's
// target_format.
[[nodiscard]] bool servesVideo(const std::filesystem::path &modelRoot);

// argv: serve-native MODEL_DIRECTORY [--hidden-states NPZ] [--keep DIRECTORY].
// The process's exit status (engine::NativeProcessExit).
[[nodiscard]] int serve(const std::filesystem::path &metallib, int argc, char **argv);

} // namespace splash::video
