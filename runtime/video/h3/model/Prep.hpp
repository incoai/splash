#pragma once

// Native prep (formerly python -m h3 prep): the request-constant DiT inputs of one (text hidden states, clip length,
// variant, seed), bit-identical to the former Python / MLX prep.
//   layout   packed rows [text L | audio 2 n | video F x 24 x 42]: tags, (t, h, w) positions (numpy float64 grids
//            of the MLX port's build_packed_sequence, cast to fp32), RoPE bf16 [M][96] = cos | sin of pos x inv_freq
//   schedule rectified-flow sigmas: linspace(1, 0, steps + 1) (torch's fp32 grid), shift 12 (video) / 3 (audio),
//            timesteps 1 - sigma, the sorted distinct timesteps (table) and per-step indices into it
//   adaLN    per block: bf16(fp32(fp64(bf16 W silu(temb) + b) + B (A silu(temb)))) for every distinct timestep (base
//            W, b from the FL2VA shards, rank-16 A / B from dit-<variant>/vdn-w8-blockNN), gathered per step and
//            modality; final layer likewise (vdn-globals LoRA); temb = time_embedder(sinusoid(table)) in fp32
//   text     condition_proj + 2-block token refiner (LoRAs merged by tools/convert_prep.py) + final RMSNorm, bf16
//   noise    MLX random.normal: threefry2x32 key sequence of mx.random.seed(seed) (one split per draw), uniform
//            on (nextafter(-1, 0), 1), erfinv, x sqrt 2; video latents [24][F][48][84] patchified, then audio rows
// GPU work runs on kernels/prep (MLX-identical arithmetic, see prep.metal); the 26 GB of adaLN bases are streamed
// from <weights>/fl2va/transformer while the GPU evaluates the previous block.
#include "video/h3/engine/Encoder.hpp"
#include "video/h3/engine/Types.hpp"
#include "video/h3/model/Geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace h3::model {

struct PrepInputs {
  const float *hidden = nullptr;  // [textRows][5120] fp32: Qwen3-VL-32B hidden_states[50], unnormalized
  int32_t textRows = 0;           // >= 2, all rows text
  int32_t steps = 0;              // 0: the variant's (turbo 8, default 50)
  uint64_t seed = 0;
};

struct Prepared {
  int32_t frames = 0, textRows = 0, rows = 0, audioRows = 0, videoRows = 0;
  int32_t latT = 0, latH = 0, latW = 0, nAud = 0, steps = 0;
  std::vector<uint8_t> tags;        // [rows]
  std::vector<uint16_t> rope;       // [rows][96] bf16: cos 48 | sin 48
  std::vector<float> pos;           // [rows][3]
  std::vector<uint16_t> text;       // [textRows][5376] bf16 refined text
  std::vector<uint16_t> adaBlocks;  // [steps][50][3][6][5376] bf16
  std::vector<uint16_t> adaFinal;   // [steps][3][2][5376] bf16
  std::vector<float> vSigmas, aSigmas;  // [steps + 1]
  std::vector<float> vT, aT;            // [steps]
  std::vector<float> table;             // distinct timesteps, sorted
  std::vector<int64_t> tv, ta;          // [steps] indices into table
  std::vector<float> videoNoise;        // [videoRows][96]
  std::vector<float> audioNoise;        // [audioRows][32]
  int32_t adalnLoraBlocks = 0;
  bool finalLora = false;
  double layoutS = 0, textS = 0, adalnS = 0, noiseS = 0, totalS = 0, adalnReadGb = 0;
};

class Prep {
public:
  static void run(metal::MetalBackend &backend, const std::string &weights, Variant variant,
                  const VideoGeometry &geometry, const PrepInputs &inputs, Prepared &out);
  // The prep directory of the former Python prep: layout.npz, sched.npz, ada_blocks.npy, ada_final.npy,
  // text_ref.npy, noise.npz, prep.json (same members, dtypes and bytes).
  static void write(const std::string &dir, const Prepared &p, Variant variant, uint64_t seed,
                    const std::string &source);
  // CPU parts (no device)
  static void schedule(int32_t steps, Prepared &p);
  static void layout(const VideoGeometry &geometry, int32_t textRows, Prepared &p);
  // RoPE of p.pos (p.rope), on the same kernels as run()
  static void rope(metal::MetalBackend &backend, Prepared &p);
  static void noiseBits(uint64_t seed, size_t videoCount, size_t audioCount, std::vector<uint32_t> &video,
                        std::vector<uint32_t> &audio);
};

} // namespace h3::model
