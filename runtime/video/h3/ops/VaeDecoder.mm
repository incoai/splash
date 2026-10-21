#include "video/h3/ops/VaeDecoder.hpp"

#include "AwakeClock.hpp"
#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/VaeABI.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace h3::ops {
namespace {

constexpr int32_t D = H3_VAE_HIDDEN, F = H3_VAE_FFN, H = H3_VAE_HEADS, HD = H3_VAE_HEAD_DIM, NP = H3_VAE_PATCH_OUT;

double secondsSince(splash::AwakeClock::time_point t0) { return splash::millisecondsSince(t0) / 1e3; }

VaeParams baseParams(int32_t rows, int32_t S, int32_t Spad) {
  VaeParams p{};
  p.M = rows;
  p.S = S;
  p.Spad = Spad;
  p.TS = H3_VAE_TILE;
  p.H = H;
  p.F = F;
  p.eps = 1e-5f;
  p.qk_scale_log2 = (1.0f / 8.0f) * 1.4426950408889634f;
  p.swizzle = device::kVaeSwizzle;
  return p;
}

} // namespace

VaeDecoder::VaeDecoder(metal::MetalBackend &backend, std::unique_ptr<model::VaeWeights> weights)
    : b_(backend), w_(std::move(weights)) {
  const size_t M = static_cast<size_t>(device::kVaeBatch) * S_;
  x_ = allocate(b_, M * D * 4, "VAE residual");
  a16_ = allocate(b_, M * D * 2, "VAE fp16 input");
  o_ = allocate(b_, M * D * 2, "VAE attention output");
  hidden_ = allocate(b_, M * F * 2, "VAE SwiGLU output");
  aq_ = allocate(b_, w_->anyInt8() ? M * F : 16, "VAE int8 input");
  sa_ = allocate(b_, M * 4, "VAE row scales");
  const size_t heads = static_cast<size_t>(device::kVaeBatch) * H * Spad_ * HD * 2;
  for (metal::MetalBuffer *t : {&q_, &k_, &v_}) {
    *t = allocate(b_, heads, "VAE Q / K / V");
    std::memset(t->contents(), 0, heads);  // pad rows must stay finite
  }
  dummy_ = allocate(b_, 256, "VAE dummy");
  bad_ = allocate(b_, 16, "VAE non-finite counter");
}

void VaeDecoder::ensure(size_t bytes, metal::MetalBuffer &buffer, const char *what) {
  if (buffer && buffer.sizeBytes() >= bytes) return;
  buffer = {};
  buffer = allocate(b_, bytes, what);
}

// X fp32 (norm) or fp16 (plain) -> fp16 y, or int8 + scale into the shared int8 input
void VaeDecoder::producer(Encoder &e, const char *kernel, const metal::MetalBuffer &x, int32_t rows, int32_t K,
                          int32_t groups, BufferView normWeight, BufferView normBias, BufferView signs,
                          const metal::MetalBuffer &y) {
  VaeParams p = baseParams(rows, S_, Spad_);
  p.K = K;
  e.pipeline(kernel);
  e.bind(x, 0);
  e.bind(aq_, 1);
  e.bind(sa_, 2);
  e.bind(y ? y : dummy_, 3);
  e.bind(normWeight ? normWeight : BufferView{dummy_}, 4);
  e.bind(normBias ? normBias : BufferView{dummy_}, 5);
  e.bind(signs ? signs : BufferView{dummy_}, 6);
  e.bytes(p, 7);
  e.dispatch(size3(rows), size3(16 * (K / 128) / groups));
}

void VaeDecoder::gemm(Encoder &e, const char *kernel, int32_t tm, const metal::MetalBuffer &a, int32_t rows, int32_t N,
                      int32_t K, BufferView w, BufferView scale, BufferView bias, const metal::MetalBuffer &y,
                      size_t yOffset, int32_t ldo, BufferView aux, BufferView aux2) {
  VaeParams p = baseParams(rows, S_, Spad_);
  p.N = N;
  p.K = K;
  p.ldo = ldo;
  e.pipeline(kernel);
  e.bind(a, 0);
  e.bind(w, 1);
  e.bind(sa_, 2);
  e.bind(scale ? scale : BufferView{dummy_}, 3);
  e.bind(bias, 4);
  e.bind({y ? y : dummy_, yOffset}, 5);
  e.bind(aux ? aux : BufferView{dummy_}, 6);
  e.bytes(p, 7);
  e.bind(q_, 8);
  e.bind(k_, 9);
  e.bind(v_, 10);
  e.bind(aux2 ? aux2 : BufferView{dummy_}, 11);
  const int32_t tn = H3_VAE_TN;
  e.dispatch(size3((N + tn - 1) / tn, (rows + tm - 1) / tm), size3(32 * H3_VAE_NSG));
}

void VaeDecoder::glu(Encoder &e, bool int8, const metal::MetalBuffer &a, int32_t rows, const model::VaeLayerWeights &w) {
  VaeParams p = baseParams(rows, S_, Spad_);
  p.N = F;
  p.K = D;
  e.pipeline(int8 ? "vae_glu_i8" : "vae_glu_f16");
  e.bind(a, 0);
  e.bind(w.fc1, 1);
  e.bind(sa_, 2);
  e.bind(w.fc1Scale ? w.fc1Scale : BufferView{dummy_}, 3);
  e.bind(w.fc1Bias, 4);
  e.bind(hidden_, 5);
  e.bytes(p, 6);
  const device::Tile t = device::kVaeTile;
  e.dispatch(size3(F / t.tn, (rows + t.tm - 1) / t.tm), size3(32 * t.nsg));
}

void VaeDecoder::attention(Encoder &e, int32_t tiles) {
  VaeParams p = baseParams(tiles * S_, S_, Spad_);
  p.ldo = D;
  const int32_t ns = device::kVaeAttentionSimdgroups;
  e.pipeline("vae_attention");
  e.bind(q_, 0);
  e.bind(k_, 1);
  e.bind(v_, 2);
  e.bind(o_, 3);
  e.bytes(p, 4);
  e.dispatch(size3((Spad_ + 16 * ns - 1) / (16 * ns), H, tiles), size3(32 * ns));
}

void VaeDecoder::embed(Encoder &e, int32_t tiles, const int32_t *origins, int32_t latentFrames, int32_t LH,
                       int32_t LW) {
  VaeParams p = baseParams(tiles * S_, S_, Spad_);
  p.LH = LH;
  p.LW = LW;
  p.T = latentFrames;
  const model::VaeGlobalWeights &g = w_->globals();
  e.pipeline("vae_embed");
  e.bind(latents_, 0);
  e.bind(g.embed, 1);
  e.bind(g.embedBias, 2);
  e.bind(g.registers, 3);
  e.bind(x_, 4);
  e.bytes(origins, 16 * size_t(tiles), 5);
  e.bytes(p, 6);
  e.dispatchThreads(size3(D / 4, tiles * S_), size3(64, 4));
}

// One transformer layer over the batch rows.
void VaeDecoder::layer(Encoder &e, int l, int32_t tiles) {
  const int32_t M = tiles * S_;
  const model::VaeLayerWeights &w = w_->layer(l);
  const model::VaeGlobalWeights &g = w_->globals();
  const device::Tile tq = device::kVaeQkvTile, t = device::kVaeTile;
  const int32_t groups = device::kVaeProducerGroups;
  // attention half
  if (w.int8Qkv()) producer(e, "vae_producer_rms_i8", x_, M, D, groups, w.norm1, {}, g.signs2048, {});
  else producer(e, "vae_producer_rms_f16", x_, M, D, groups, w.norm1, {}, {}, a16_);
  gemm(e, w.int8Qkv() ? "vae_qkv_i8" : "vae_qkv_f16", tq.tm, w.int8Qkv() ? aq_ : a16_, M, 3 * D, D, w.qkv,
       w.qkvScale, w.qkvBias, {}, 0, 0, g.cos, g.sin);
  attention(e, tiles);
  if (w.int8Out()) producer(e, "vae_producer_i8", o_, M, D, groups, {}, {}, g.signs2048, {});
  gemm(e, w.int8Out() ? "vae_residual_i8" : "vae_residual_f16", t.tm, w.int8Out() ? aq_ : o_, M, D, D, w.out,
       w.outScale, w.outBias, x_, 0, D, w.scale1, {});
  // MLP half
  if (w.int8Fc1()) producer(e, "vae_producer_rms_i8", x_, M, D, groups, w.norm2, {}, g.signs2048, {});
  else producer(e, "vae_producer_rms_f16", x_, M, D, groups, w.norm2, {}, {}, a16_);
  glu(e, w.int8Fc1(), w.int8Fc1() ? aq_ : a16_, M, w);
  if (w.int8Fc2())
    producer(e, "vae_producer_ffn_i8", hidden_, M, F, device::kVaeFfnProducerGroups, {}, {}, g.signs8192, {});
  gemm(e, w.int8Fc2() ? "vae_residual_i8" : "vae_residual_f16", t.tm, w.int8Fc2() ? aq_ : hidden_, M, D, F, w.fc2,
       w.fc2Scale, w.fc2Bias, x_, 0, D, w.scale2, {});
}

void VaeDecoder::head(Encoder &e, int32_t tiles, size_t outRow0) {
  const int32_t M = tiles * S_;
  const model::VaeGlobalWeights &g = w_->globals();
  producer(e, "vae_producer_ln_f16", x_, M, D, device::kVaeProducerGroups, g.lnWeight, g.lnBias, {}, a16_);
  gemm(e, "vae_proj_f16", device::kVaeTile.tm, a16_, M, NP, D, g.proj, {}, g.projBias, out_,
       outRow0 * NP * sizeof(float), NP, {}, {});
}

void VaeDecoder::decode(const float *latents, int32_t latentFrames, const model::VideoGeometry &geometry,
                        const model::VaePlan &plan, uint8_t *frames, DecodeStats &stats) {
  const auto tStart = splash::AwakeClock::now();
  stats = DecodeStats{};
  const int32_t LH = geometry.latentHeight, LW = geometry.latentWidth;
  const int32_t HP = LH * 16, WP = LW * 16;
  const int32_t nty = static_cast<int32_t>(plan.ys.size()), ntx = static_cast<int32_t>(plan.xs.size());
  const int32_t tiles = nty * ntx, CF = 4 * plan.ct, NF0 = CF - plan.fpp, fov = plan.fov;
  const int64_t outFrames = plan.outputFrames();
  if (nty > H3_VAE_MAX_TILES || ntx > H3_VAE_MAX_TILES || fov > NF0 || fov != 4 * plan.tov - plan.fpp)
    throw std::invalid_argument("unsupported VAE decode plan");
  ensure(static_cast<size_t>(tiles) * S_ * NP * sizeof(float), out_, "VAE proj_out tokens");
  const size_t fsz = static_cast<size_t>(HP) * WP * 3;
  for (int i = 0; i < 2; ++i) {
    ensure(fsz * fov * 4, overlap_[i], "VAE overlap frames");
    ensure(fsz * NF0, stage_[i], "VAE frame staging");
  }
  const size_t latentBytes = static_cast<size_t>(latentFrames) * LH * LW * H3_VAE_LATENT * 4;
  ensure(latentBytes, latents_, "VAE latents");
  std::memcpy(latents_.contents(), latents, latentBytes);
  stats.uploadS = secondsSince(tStart);
  std::memset(bad_.contents(), 0, 16);
  VaeFinishParams fp{};
  fp.NTY = nty;
  fp.NTX = ntx;
  fp.HP = HP;
  fp.WP = WP;
  fp.S = S_;
  fp.TS = H3_VAE_TILE;
  fp.NF0 = NF0;
  fp.FPP = plan.fpp;
  fp.FOV = fov;
  fp.CF = CF;
  for (int32_t i = 0; i < nty; ++i) {
    fp.ys[i] = plan.ys[i];
    fp.ovy[i] = i < nty - 1 ? plan.ovy[i] : 0;
  }
  for (int32_t j = 0; j < ntx; ++j) {
    fp.xs[j] = plan.xs[j];
    fp.ovx[j] = j < ntx - 1 ? plan.ovx[j] : 0;
  }
  static_assert(sizeof(fp.pm) == 4 * sizeof(float));
  // pixel de-normalisation (minimax_h3_mlx.packing PIXEL_MEAN / PIXEL_STD)
  const double pixelMean[3] = {0.485, 0.456, 0.406}, pixelStd[3] = {0.229, 0.224, 0.225};
  for (int c = 0; c < 3; ++c) {
    fp.pm[c] = static_cast<float>(pixelMean[c]);
    fp.ps[c] = static_cast<float>(pixelStd[c]);
  }
  double copyS = 0;
  // frames [first, first + count) of a staging buffer, clipped to the output (pad frames are cut)
  auto copyFrames = [&](int slot, int64_t first, int64_t count) {
    const auto t0 = splash::AwakeClock::now();
    const int64_t n = std::min<int64_t>(count, outFrames - first);
    if (n > 0) std::memcpy(frames + first * fsz, stage_[slot].contents(), fsz * n);
    copyS += secondsSince(t0);
  };
  const auto gpuStart = splash::AwakeClock::now();
  metal::CommandTicket previous;
  for (int32_t c = 0; c < plan.chunks; ++c) {
    // the clip's command: its batches of tile-clips, then its finish
    Encoder e(b_);
    const int32_t t0 = c * plan.ct;
    for (int32_t k0 = 0; k0 < tiles; k0 += device::kVaeBatch) {
      const int32_t nb = std::min(device::kVaeBatch, tiles - k0);
      std::vector<int32_t> origins(4 * nb);
      for (int32_t i = 0; i < nb; ++i) {
        const int32_t ti = (k0 + i) / ntx, tj = (k0 + i) % ntx;
        origins[4 * i] = t0;
        origins[4 * i + 1] = plan.ys[ti] / 16;
        origins[4 * i + 2] = plan.xs[tj] / 16;
        origins[4 * i + 3] = 0;
      }
      embed(e, nb, origins.data(), latentFrames, LH, LW);
      for (int l = 0; l < H3_VAE_LAYERS; ++l) layer(e, l, nb);
      head(e, nb, static_cast<size_t>(k0) * S_);
    }
    fp.has_prev = c > 0;
    fp.mode = 0;
    e.pipeline("vae_finish");
    e.bind(out_, 0);
    e.bind(overlap_[(c + 1) % 2], 1);
    e.bind(overlap_[c % 2], 2);
    e.bind(stage_[c % 2], 3);
    e.bind(bad_, 4);
    e.bytes(fp, 5);
    e.dispatchThreads(size3(WP, HP, NF0 + fov), size3(32, 8));
    // one command in flight: clip c starts once clip c - 1 completed, whose frames are copied while c runs
    if (c > 0) stats.gpuS += wait(previous);
    previous = submit(b_, e);
    if (c > 0) copyFrames((c - 1) % 2, int64_t(c - 1) * NF0, NF0);
  }
  stats.gpuS += wait(previous);
  copyFrames((plan.chunks - 1) % 2, int64_t(plan.chunks - 1) * NF0, NF0);
  {  // tail: the last clip's overlap frames
    Encoder e(b_);
    fp.mode = 1;
    e.pipeline("vae_finish");
    e.bind(out_, 0);
    e.bind(overlap_[(plan.chunks - 1) % 2], 1);
    e.bind(dummy_, 2);
    e.bind(stage_[0], 3);
    e.bind(bad_, 4);
    e.bytes(fp, 5);
    e.dispatchThreads(size3(WP, HP, fov), size3(32, 8));
    stats.gpuS += run(b_, e);
    copyFrames(0, int64_t(plan.chunks) * NF0, fov);
  }
  stats.gpuSpanS = secondsSince(gpuStart);
  stats.wallS = secondsSince(tStart);
  stats.hostCopyS = copyS;
  stats.nonfinite = static_cast<const uint32_t *>(bad_.contents())[0];
  stats.deviceMb = deviceMb(b_);
}

} // namespace h3::ops
