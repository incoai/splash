#pragma once

// Native prep (formerly python -m h3 prep): the request-constant DiT inputs of one (text hidden states, clip length,
// variant, seed, optional FL2VA keyframes), bit-identical to the former Python / MLX prep for text-to-video.
//   layout   packed rows [text L | keyframes C | audio 2 n | video F x 24 x 42]: tags, (t, h, w) positions (numpy
//            float64 grids of the MLX port's build_packed_sequence, cast to fp32; a keyframe sits on the frame grid at
//            the time of the first latent frame, or of the last one for a last frame), RoPE bf16 [M][96] = cos | sin
//            of pos x inv_freq
//   schedule rectified-flow sigmas: linspace(1, 0, steps + 1) (torch's fp32 grid), shift 12 (video) / 3 (audio),
//            timesteps 1 - sigma, the sorted distinct timesteps (table) and per-step indices into it
//   adaLN    per block: bf16(fp32(fp64(bf16 W silu(temb) + b) + B (A silu(temb)))) for every distinct timestep (base
//            W, b from the FL2VA shards, rank-16 A / B from dit-<variant>/vdn-w8-blockNN), gathered per step and
//            modality; final layer likewise (vdn-globals LoRA); temb = time_embedder(sinusoid(table)) in fp32. The
//            tables depend only on the variant's schedule: tabulateAdaLn computes them once, at conversion, into
//            dit-<variant>/adaln.safetensors, which run() gathers from
//   text     condition_proj + 2-block token refiner (LoRAs merged by tools/convert_prep.py) + final RMSNorm, bf16
//   noise    MLX random.normal: threefry2x32 key sequence of mx.random.seed(seed) (one split per draw), uniform
//            on (nextafter(-1, 0), 1), erfinv, x sqrt 2; keyframe noise [C][96] (FL2VA), video latents
//            [24][F][48][84] patchified, then audio rows
//   keyframes (FL2VA, pipeline._encode_keyframes) the video VAE encoder's posterior moments of each keyframe
//            (ops/VaeEncoder), sampled under mx.random.seed(42), float16 round trip, normalised, patchified, then
//            noised to t = 0.999 with the request's keyframe noise: rows that stay fixed through the denoising loop
//            and run at timestep max(t, 0.999) (adaLN slot 3 = the video modulation at that timestep)
// GPU work runs on kernels/prep (MLX-identical arithmetic, see prep.metal). A request reads 0.15 GB (turbo) / 1 GB
// (default) of adaLN tables; their conversion streams the 26 GB of adaLN bases from FL2VA's transformer shards while
// the GPU evaluates the previous block.
#include "video/metal/Encoder.hpp"
#include "video/ops/Geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace splash::video::model {

// The DiT variants: turbo (8 steps), default (50).
enum class Variant : int32_t { Turbo = 0, Default = 1 };

struct PrepInputs {
  const float *hidden = nullptr;  // [textRows][5120] fp32: Qwen3-VL-32B hidden_states[50], unnormalized
  int32_t textRows = 0;           // >= 2
  int32_t steps = 0;              // 0 or the variant's (turbo 8, default 50): the schedule of its adaLN tables
  uint64_t seed = 0;
  const uint8_t *textTags = nullptr;  // [textRows] 1 text, 0 video (rows of a keyframe's vision block); null: all 1
  // FL2VA keyframes: uint8 RGB [height][width][3] of the request canvas (resized / cropped by the host as the port's
  // prepare_keyframe_image), null when absent
  const uint8_t *firstFrame = nullptr, *lastFrame = nullptr;
};

struct Prepared {
  int32_t frames = 0, textRows = 0, rows = 0, audioRows = 0, videoRows = 0;
  int32_t latT = 0, latH = 0, latW = 0, nAud = 0, steps = 0;
  int32_t condRows = 0;             // keyframes x 1008 (0: text-to-video)
  std::vector<int32_t> anchors;     // per keyframe in packed order: 0 first, 1 last
  std::vector<uint8_t> tags;        // [rows] 0 video, 1 text, 2 audio, 3 conditioning video
  std::vector<uint16_t> rope;       // [rows][96] bf16: cos 48 | sin 48
  std::vector<float> pos;           // [rows][3]
  std::vector<uint16_t> text;       // [textRows][5376] bf16 refined text
  std::vector<uint16_t> adaBlocks;  // [steps][50][adaModalities()][6][5376] bf16
  std::vector<uint16_t> adaFinal;   // [steps][3][2][5376] bf16
  std::vector<float> vSigmas, aSigmas;  // [steps + 1]
  std::vector<float> vT, aT;            // [steps]
  std::vector<float> table;             // distinct timesteps, sorted
  std::vector<int64_t> tv, ta;          // [steps] indices into table
  std::vector<float> cT;                // [steps] conditioning timestep max(t, 0.999) (keyframes only)
  std::vector<int64_t> tc;
  std::vector<float> videoNoise;        // [videoRows][96]
  std::vector<float> audioNoise;        // [audioRows][32]
  std::vector<float> moments;           // [keyframes][latH][latW][48] encoder posterior moments
  std::vector<float> condClean;         // [condRows][96] normalised keyframe latents
  std::vector<float> cond;              // [condRows][96] noised to t = 0.999: the fixed DiT inputs
  int32_t adalnLoraBlocks = 0;
  bool finalLora = false;
  double layoutS = 0, textS = 0, adalnS = 0, noiseS = 0, totalS = 0, adalnReadGb = 0, encodeS = 0;
  [[nodiscard]] int32_t keyframes() const { return static_cast<int32_t>(anchors.size()); }
  [[nodiscard]] int32_t adaModalities() const { return condRows ? 4 : 3; }
};

class Prep {
public:
  static void run(metal::MetalBackend &backend, const std::string &weights, Variant variant,
                  const ops::VideoGeometry &geometry, const PrepInputs &inputs, Prepared &out);
  // Conversion: the adaLN tables of the variant's schedule, without keyframes and with them (adaln.text.* and
  // adaln.keyframes.*: table F32 [T], final BF16 [T][2][5376], blocks BF16 [50][T][6 x 3 x 5376]), from the converted
  // DiT in ditDir and the adaLN bases of FL2VA's transformer shards in transformerDir, into ditDir/adaln.safetensors.
  static void tabulateAdaLn(metal::MetalBackend &backend, const std::string &ditDir, const std::string &transformerDir,
                            Variant variant);
  // CPU parts (no device)
  static void schedule(int32_t steps, bool conditioned, Prepared &p);
  // anchors: per keyframe 0 first / 1 last; textTags nullable (all text)
  static void layout(const ops::VideoGeometry &geometry, int32_t textRows, const uint8_t *textTags,
                     const std::vector<int32_t> &anchors, Prepared &p);
  static void layout(const ops::VideoGeometry &geometry, int32_t textRows, Prepared &p) {
    layout(geometry, textRows, nullptr, {}, p);
  }
  // RoPE of p.pos (p.rope), on the same kernels as run()
  static void rope(metal::MetalBackend &backend, Prepared &p);
  static void noiseBits(uint64_t seed, size_t videoCount, size_t audioCount, std::vector<uint32_t> &video,
                        std::vector<uint32_t> &audio);
  // consecutive draws of mx.random.seed(seed): one key split per draw
  static void drawBits(uint64_t seed, const std::vector<size_t> &counts, std::vector<std::vector<uint32_t>> &out);
};

} // namespace splash::video::model
