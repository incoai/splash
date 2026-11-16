#pragma once

// Request preparation: the request-constant DiT inputs of one request (text hidden states, clip length, variant, seed,
// optional FL2VA keyframes), bit-identical to the MLX port's prep for text-to-video.
//   layout   packed rows [text L | keyframes C | audio 2 n | video F x G_H x G_W] as row classes (ops/RowLayout):
//            tags, (t, h, w) positions (numpy float64 grids of the MLX port's build_packed_sequence, cast to fp32; a
//            keyframe sits on the frame grid at the time of the first latent frame, or of the last one for a last
//            frame), RoPE bf16 [M][96] = cos | sin of pos x inv_freq. Ref2VA's reference rows (referenceRows) use the
//            same classes, in the engine's row order (ops::RowLayout::engineOrder)
//   schedule rectified-flow sigmas: linspace(1, 0, steps + 1) (torch's fp32 grid), shift 12 (video) / 3 (audio),
//            timesteps 1 - sigma, the sorted distinct timesteps (table) and per-step indices into it
//   adaLN    (slots 0 video, 1 text, 2 audio, 3 conditioning video at max(t, 0.999), 4 conditioning audio at t = 1)
//            per block: bf16(fp32(fp64(bf16 W silu(temb) + b) + B (A silu(temb)))) for every distinct timestep (base
//            W, b from the FL2VA shards, rank-16 A / B from dit-<variant>/vdn-w8-blockNN), gathered per step and
//            modality; final layer likewise; temb = time_embedder(sinusoid(table)) on the host (timestepEmbedding). The
//            tables depend only on the variant's schedule: tabulateAdaLn computes them once, at conversion, into
//            dit-<variant>/adaln.safetensors, which run() gathers from
//   text     condition_proj + 2-block token refiner (LoRAs merged at installation; a runtime adapter's added as PEFT
//            does, bf16 linears y = W x + B (A x)) + final RMSNorm, bf16
//   noise    MLX random.normal: threefry2x32 key sequence of mx.random.seed(seed) (one split per draw), uniform
//            on (nextafter(-1, 0), 1), erfinv, x sqrt 2; keyframe noise [C][96] (FL2VA), video latents
//            [24][F][H / 16][W / 16] patchified, then audio rows
//   keyframes (FL2VA, pipeline._encode_keyframes) the video VAE encoder's posterior moments of each keyframe
//            (ops/VaeEncoder), sampled under mx.random.seed(42), float16 round trip, normalised, patchified, then
//            noised to t = 0.999 with the request's keyframe noise: rows that stay fixed through the denoising loop
//            and run at timestep max(t, 0.999) (adaLN slot 3 = the video modulation at that timestep)
// GPU work runs on kernels/video/prep (MLX-identical arithmetic, see prep.metal). A request reads 0.15 GB (turbo) / 1 GB
// (default) of adaLN tables; their conversion streams the 26 GB of adaLN bases from FL2VA's transformer shards while
// the GPU evaluates the previous block.
#include "video/metal/Encoder.hpp"
#include "video/model/Adapter.hpp"
#include "video/ops/Geometry.hpp"
#include "video/ops/RowLayout.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace splash::video::model {

struct FastH3Contract;

// A Ref2VA reference with its latents (video/model/References.hpp computes them from the media, or a dump supplies them):
// video / image the video VAE's normalized latents [24][F][h][w] (posterior sample, float16 round trip, normalized as
// pipeline._encode_keyframes), audio the audio VAE's latents [T][64] (per latent: channel 0's 32 | channel 1's 32), a
// video's soundtrack or an audio reference.
struct ReferenceInput {
  ops::ReferenceShape shape;
  const float *video = nullptr;
  const float *audio = nullptr;
};

struct PrepInputs {
  const float *hidden = nullptr;  // [textRows][5120] fp32: Qwen3-VL-32B hidden_states[50], unnormalized
  int32_t textRows = 0;           // >= 2
  int32_t steps = 0;              // its adaLN tables': a VDN-H3 DiT's (> 0), a FastH3 DiT's rungs' (or 0)
  uint64_t seed = 0;
  const uint8_t *textTags = nullptr;  // [textRows] 1 text, 0 video (rows of a keyframe's vision block); null: all 1
  // FL2VA keyframes: uint8 RGB [height][width][3] of the request canvas (resized / cropped by the host as the port's
  // prepare_keyframe_image), null when absent
  const uint8_t *firstFrame = nullptr, *lastFrame = nullptr;
  // Runtime adapters (video/model/Adapter.hpp): the token refiner's LoRAs run in the text path; at most one
  std::vector<AdapterSpec> adapters;
  // Ref2VA references in presentation order (Prep::referenceRows), and the augmentation noise of their video-stream
  // rows, [condition video rows][96] (null: referenceRows leaves them clean; run() draws it as the keyframes' noise,
  // the request's first draw)
  std::vector<ReferenceInput> references;
  const float *referenceNoise = nullptr;
};

struct Prepared {
  int32_t frames = 0, textRows = 0, rows = 0, audioRows = 0, videoRows = 0;
  int32_t latT = 0, latH = 0, latW = 0, nAud = 0, steps = 0;
  int32_t condRows = 0;             // keyframes x rows per latent frame (0: text-to-video)
  std::vector<uint8_t> tags;        // [rows] 0 video, 1 text, 2 audio, 3 conditioning video
  std::vector<uint16_t> rope;       // [rows][96] bf16: cos 48 | sin 48
  std::vector<float> pos;           // [rows][3]
  std::vector<uint16_t> text;       // [textRows][5376] bf16 refined text
  std::vector<uint16_t> adaBlocks;  // [steps][50][adaModalities()][6][5376] bf16
  std::vector<uint16_t> adaFinal;   // [steps][finalSlots()][2][5376] bf16
  std::vector<float> vSigmas, aSigmas;  // [steps + 1]
  std::vector<float> vT, aT;            // [steps]
  std::vector<float> table;             // distinct timesteps, sorted
  std::vector<int64_t> tv, ta;          // [steps] indices into table
  std::vector<float> cT;                // [steps] conditioning timestep max(t, 0.999) (keyframes, references)
  std::vector<int64_t> tc;
  std::vector<int64_t> tk;              // [steps] index of the clean timestep 1.0 (references)
  bool references = false;              // Ref2VA: sequence's conditioning rows, cond and condAudio
  std::vector<float> videoNoise;        // [videoRows][96]
  std::vector<float> audioNoise;        // [audioRows][32]
  std::vector<float> moments;           // [keyframes][latH][latW][48] encoder posterior moments
  std::vector<float> cond;              // [condRows][96] noised to t = 0.999: the fixed DiT inputs (referenceRows:
                                        // [sequence.condVideoRows][96], every video-stream condition row)
  std::vector<float> condAudio;         // referenceRows: [sequence.condAudioRows][32] clean audio condition rows
  ops::RowLayout sequence;              // the row classes of the packed rows (float64 positions, indices, metadata)
  // Sampler "renoise" (pika-6, video/model/FastH3.hpp): the fresh noise of steps 0 .. steps - 2, MLX random.normal draws of
  // the stream seed + renoiseSeedOffset, per step video ([24][F][H][W] patchified as videoNoise) then audio
  bool renoise = false;
  uint64_t renoiseSeedOffset = 1;
  std::vector<float> renoiseVideo, renoiseAudio;  // [steps - 1][videoRows][96], [steps - 1][audioRows][32]
  double layoutS = 0, textS = 0, adalnS = 0, totalS = 0, adalnReadGb = 0, encodeS = 0;
  [[nodiscard]] int32_t adaModalities() const {
    return sequence.condAudioRows ? 5 : (condRows || sequence.condVideoRows ? 4 : 3);
  }
  [[nodiscard]] int32_t finalSlots() const { return sequence.condAudioRows ? 5 : 3; }
};

class Prep {
public:
  // dit: the DiT's name, its directory weights/dit-<dit> (model::ditDirectory)
  static void run(metal::MetalBackend &backend, const std::string &weights, const std::string &dit,
                  const ops::VideoGeometry &geometry, const PrepInputs &inputs, Prepared &out);
  // Conversion: the adaLN tables of the DiT's schedule (a VDN-H3 DiT's of `steps`, a FastH3 DiT's of its rungs, steps
  // 0), without keyframes and with them (adaln.text.* and
  // adaln.keyframes.*: table F32 [T], final BF16 [T][2][5376], blocks BF16 [50][T][6 x 3 x 5376]) and, for a FastH3
  // DiT whose transformerDir holds blocks.N.adaln.clean (t = 1), with references (adaln.references.*), from the converted
  // DiT in ditDir and the adaLN bases of FL2VA's transformer shards in transformerDir, into ditDir/adaln.safetensors.
  // FastH3's blocks take their tables of the text schedule from ditDir's block files and the rows of the keyframe
  // timestep, blocks.N.adaln.keyframe, from transformerDir (written at installation); the final layer's are computed.
  static void tabulateAdaLn(metal::MetalBackend &backend, const std::string &ditDir, const std::string &transformerDir,
                            int32_t steps);
  // CPU parts (no device)
  // silu(time_embedder(sinusoid(t))) of each timestep of t, as the official runs' fp32 module computes it: fp32 values,
  // fp64 accumulation; st fp32 [T][2688], stBf its bf16 (RNE) bits. w1 [5376][256], b1, w2 [2688][5376], b2: the DiT's
  // time_embedder.proj_in / proj_out.
  static void timestepEmbedding(const std::vector<float> &t, const std::vector<float> &w1, const std::vector<float> &b1,
                                const std::vector<float> &w2, const std::vector<float> &b2, float *st, uint16_t *stBf);
  static void schedule(int32_t steps, bool conditioned, Prepared &p);
  // FastH3: one step per trained rung, in place of schedule() (video/model/FastH3.mm)
  static void scheduleRungs(const FastH3Contract &contract, bool conditioned, Prepared &p);
  // anchors: per keyframe 0 first / 1 last; textTags nullable (all text)
  static void layout(const ops::VideoGeometry &geometry, int32_t textRows, const uint8_t *textTags,
                     const std::vector<int32_t> &anchors, Prepared &p);
  static void layout(const ops::VideoGeometry &geometry, int32_t textRows, Prepared &p) {
    layout(geometry, textRows, nullptr, {}, p);
  }
  // Ref2VA on the CPU: the layout of in.textRows text rows (in.textTags) and in.references, and the fixed rows' initial
  // state: p.cond the video-stream condition rows (patchified, then x' = 0.999 x + (1 - 0.999) eps with
  // in.referenceNoise, fp32 as keyframes), p.condAudio the audio condition rows (clean, channel-major). The target rows
  // start from the request's noise (videoNoise, audioNoise). Throws std::invalid_argument for missing latents or
  // keyframes (their pixels need the GPU encoder).
  static void referenceRows(const ops::VideoGeometry &geometry, const PrepInputs &inputs, Prepared &p);
  // RoPE of p.pos (p.rope), on the same kernels as run()
  static void rope(metal::MetalBackend &backend, Prepared &p);
  static void noiseBits(uint64_t seed, size_t videoCount, size_t audioCount, std::vector<uint32_t> &video,
                        std::vector<uint32_t> &audio);
  // consecutive draws of mx.random.seed(seed): one key split per draw
  static void drawBits(uint64_t seed, const std::vector<size_t> &counts, std::vector<std::vector<uint32_t>> &out);
};

} // namespace splash::video::model
