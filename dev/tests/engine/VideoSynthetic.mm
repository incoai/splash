#include "VideoSynthetic.hpp"

#include "metal/abi/video/Layout.h"

#include <cmath>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace splash::video::test {

uint64_t syntheticSeed(int group, int id) {
  return (uint64_t(h3_syn_hash(0x48335359ull, uint64_t(group))) << 32) | h3_syn_hash(0x48335359ull + 1, uint64_t(id));
}

float syntheticFloat(uint64_t seed, uint64_t i, float scale, float offset) {
  const float r = static_cast<float>(static_cast<int32_t>(h3_syn_hash(seed, i)) >> 8) * 0x1p-23f;
  return std::fma(scale, r, offset);
}

void syntheticF32(uint64_t seed, size_t n, float scale, float offset, float *out) {
  for (size_t i = 0; i < n; ++i) out[i] = syntheticFloat(seed, i, scale, offset);
}

void syntheticBf16(uint64_t seed, size_t n, float scale, float offset, uint16_t *out) {
  for (size_t i = 0; i < n; ++i) {
    const float f = syntheticFloat(seed, i, scale, offset);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    out[i] = static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
  }
}

metal::MetalBuffer SyntheticFiller::allocate(size_t count, int type, uint64_t seed, float scale, float offset,
                                             std::string_view what) {
  const size_t elem = type == H3_SYN_I8 ? 1 : type == H3_SYN_F32 ? 4 : 2;
  metal::MetalBuffer out = video::allocate(b_, count * elem, what);
  fill(out, 0, count, type, seed, scale, offset);
  return out;
}

void SyntheticFiller::fill(const metal::MetalBuffer &buffer, size_t offsetBytes, size_t count, int type, uint64_t seed,
                           float scale, float offset) {
  const size_t elem = type == H3_SYN_I8 ? 1 : type == H3_SYN_F32 ? 4 : 2;
  jobs_.push_back({buffer, offsetBytes,
                   SyntheticFill{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                                 static_cast<uint32_t>(count), type, scale, offset}});
  bytes_ += count * elem;
}

void SyntheticFiller::run() {
  Encoder e(b_);
  e.pipeline("synthetic_fill");
  for (const Job &j : jobs_) {
    e.bind({j.buffer, j.offset}, 0);
    e.bytes(j.fill, 1);
    e.dispatchThreads(size3(j.fill.count), size3(256));
  }
  video::run(b_, e);
  jobs_.clear();
}

std::unique_ptr<model::DitWeights> syntheticDitWeights(metal::MetalBackend &backend, const std::set<int> &qkfixBlocks,
                                                      const std::set<int> &blocks) {
  constexpr int64_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER, QKV = H3_DIT_QKV, G = H3_DIT_GATES;
  constexpr int64_t HEADS = H3_DIT_HEADS, HD = H3_DIT_HEAD_DIM;
  std::vector<model::DitBlockWeights> blockWeights(H3_DIT_BLOCKS);
  std::map<int, model::QkFixWeights> qkfix;
  model::DitIoWeights o;
  SyntheticFiller f(backend);
  // int8 values are uniform on [-127, 127] (std 73.3): row scales 1 / (73.3 sqrt(K)) x (1 +- 0.25) give unit gain
  auto rowScale = [](int64_t K) { return static_cast<float>(1.0 / (73.32 * std::sqrt(double(K)))); };
  auto uniform = [](int64_t fanIn) { return static_cast<float>(std::sqrt(3.0 / double(fanIn))); };
  for (int b = 0; b < H3_DIT_BLOCKS; ++b) {
    if (!blocks.empty() && !blocks.count(b)) continue;
    model::DitBlockWeights &x = blockWeights[b];
    auto seed = [b](int id) { return syntheticSeed(b, id); };
    x.qkvg = f.allocate(size_t((QKV + G) * D), H3_SYN_I8, seed(0), 0, 0, "synthetic qkvg");
    x.qkvgScale = f.allocate(size_t(QKV + G), H3_SYN_F32, seed(1), 0.25f * rowScale(D), rowScale(D), "scale");
    x.out = f.allocate(size_t(D * C), H3_SYN_I8, seed(2), 0, 0, "synthetic out");
    x.outScale = f.allocate(size_t(D), H3_SYN_F32, seed(3), 0.25f * rowScale(C), rowScale(C), "scale");
    x.fc1 = f.allocate(size_t(2 * F * D), H3_SYN_I8, seed(4), 0, 0, "synthetic fc1");
    x.fc1Scale = f.allocate(size_t(2 * F), H3_SYN_F32, seed(5), 0.25f * rowScale(D), rowScale(D), "scale");
    x.fc2 = f.allocate(size_t(D * F), H3_SYN_I8, seed(6), 0, 0, "synthetic fc2");
    x.fc2Scale = f.allocate(size_t(D), H3_SYN_F32, seed(7), 0.25f * rowScale(F), rowScale(F), "scale");
    x.outLinear = f.allocate(size_t(D * C), H3_SYN_I8, seed(8), 0, 0, "synthetic out_linear");
    x.outLinearScale = f.allocate(size_t(D), H3_SYN_F32, seed(9), 0.25f * rowScale(C), rowScale(C), "scale");
    x.norm1 = f.allocate(size_t(D), H3_SYN_BF16, seed(10), 0.1f, 1.0f, "norm1");
    x.norm2 = f.allocate(size_t(D), H3_SYN_BF16, seed(11), 0.1f, 1.0f, "norm2");
    x.qNorm = f.allocate(size_t(HD), H3_SYN_BF16, seed(12), 0.1f, 1.0f, "q_norm");
    x.kNorm = f.allocate(size_t(HD), H3_SYN_BF16, seed(13), 0.1f, 1.0f, "k_norm");
    x.sgBias = f.allocate(size_t(HEADS), H3_SYN_F32, seed(14), 0.5f, 0.0f, "sg_bias");
    x.ogUp = f.allocate(size_t(C * HD), H3_SYN_F16, seed(15), uniform(HD), 0.0f, "og_up");
    x.ogBias = f.allocate(size_t(C), H3_SYN_F32, seed(16), 0.1f, 0.0f, "og_bias");
    x.alphaDown = f.allocate(size_t(HD * D), H3_SYN_F32, seed(17), uniform(D), 0.0f, "alpha_down");
    x.alphaUp = f.allocate(size_t(C * HD), H3_SYN_F32, seed(18), uniform(HD), 0.0f, "alpha_up");
    x.dtBias = f.allocate(size_t(C), H3_SYN_F32, seed(19), 0.5f, 0.0f, "dt_bias");
    x.aLog = f.allocate(size_t(HEADS), H3_SYN_F32, seed(20), 0.5f, 0.0f, "A_log");
    x.kSpatial = f.allocate(size_t(25 * C), H3_SYN_F16, seed(21), 0.2f, 0.0f, "k_sp");
    x.kTemporal = f.allocate(size_t(5 * C), H3_SYN_F16, seed(22), 0.2f, 0.0f, "k_tm");
    x.vSpatial = f.allocate(size_t(25 * C), H3_SYN_F16, seed(23), 0.2f, 0.0f, "v_sp");
    x.vTemporal = f.allocate(size_t(5 * C), H3_SYN_F16, seed(24), 0.2f, 0.0f, "v_tm");
    x.branchNorm = f.allocate(size_t(HD), H3_SYN_F32, seed(25), 0.1f, 1.0f, "branch norm");
    if (qkfixBlocks.count(b)) {
      model::QkFixWeights &q = qkfix[b];
      q.wqkv = f.allocate(size_t(QKV * D), H3_SYN_F16, seed(26), uniform(D), 0.0f, "qkfix wqkv");
      const int32_t split[4] = {0, 48, -1, -1};
      q.splitDims = upload(backend, split, sizeof split, "qkfix split");
    }
  }
  const int64_t KV = H3_DIT_VIDEO_CHANNELS, KA = H3_DIT_AUDIO_CHANNELS;
  auto io = [](int id) { return syntheticSeed(kSyntheticIoGroup, id); };
  o.videoProj = f.allocate(size_t(KV * D), H3_SYN_F32, io(0), uniform(KV), 0.0f, "video_patch_proj");
  o.videoProjBias = f.allocate(size_t(D), H3_SYN_F32, io(1), 0.1f, 0.0f, "video_patch_proj.bias");
  o.audioProj = f.allocate(size_t(KA * D), H3_SYN_F32, io(2), uniform(KA), 0.0f, "audio_patch_proj");
  o.audioProjBias = f.allocate(size_t(D), H3_SYN_F32, io(3), 0.1f, 0.0f, "audio_patch_proj.bias");
  o.finalNorm = f.allocate(size_t(D), H3_SYN_BF16, io(4), 0.1f, 1.0f, "final norm");
  o.videoOut = f.allocate(size_t(KV * D), H3_SYN_F32, io(5), uniform(D), 0.0f, "video_out");
  o.videoOutBias = f.allocate(size_t(KV), H3_SYN_F32, io(6), 0.1f, 0.0f, "video_out.bias");
  o.audioOut = f.allocate(size_t(KA * D), H3_SYN_F32, io(7), uniform(D), 0.0f, "audio_out");
  o.audioOutBias = f.allocate(size_t(KA), H3_SYN_F32, io(8), 0.1f, 0.0f, "audio_out.bias");
  f.run();
  return std::make_unique<model::DitWeights>(std::move(blockWeights), std::move(qkfix), std::move(o), f.bytes());
}

} // namespace splash::video::test
