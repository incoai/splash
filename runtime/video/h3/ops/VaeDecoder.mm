#include "video/h3/ops/VaeDecoder.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/VaeABI.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace h3::ops {
namespace {

constexpr int32_t D = H3_VAE_HIDDEN, F = H3_VAE_FFN, H = H3_VAE_HEADS, HD = H3_VAE_HEAD_DIM, NP = H3_VAE_PATCH_OUT;

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

const char *const kStageNames[VaeDecoder::kStages] = {"embed", "norm", "qkv", "attn", "quant", "out",
                                                      "fc1",   "fc2",  "ln",  "proj", "finish"};

} // namespace

// One command buffer for a batch, or (profile) one command buffer per dispatch with GPU time per stage.
class VaeDecoder::Recorder {
public:
  explicit Recorder(VaeDecoder &d) : d_(d) {}
  id<MTLComputeCommandEncoder> begin(int stage) {
    stage_ = stage;
    if (!cb_) {
      cb_ = [d_.b_.queue() commandBuffer];
      encoder_ = [cb_ computeCommandEncoder];
    }
    return encoder_;
  }
  void end() {
    if (!d_.profile_) return;
    [encoder_ endEncoding];
    [cb_ commit];
    [cb_ waitUntilCompleted];
    d_.stageMs_[stage_] += (cb_.GPUEndTime - cb_.GPUStartTime) * 1e3;
    d_.stageDispatches_[stage_] += 1;
    cb_ = nil;
    encoder_ = nil;
  }
  id<MTLCommandBuffer> commit() {
    if (!cb_) return nil;
    [encoder_ endEncoding];
    [cb_ commit];
    id<MTLCommandBuffer> c = cb_;
    cb_ = nil;
    encoder_ = nil;
    return c;
  }

private:
  VaeDecoder &d_;
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLComputeCommandEncoder> encoder_ = nil;
  int stage_ = 0;
};

const char *VaeDecoder::stageName(int stage) { return stage >= 0 && stage < kStages ? kStageNames[stage] : ""; }

Status VaeDecoder::create(metal::MetalBackend &backend, std::unique_ptr<model::VaeWeights> weights,
                          std::unique_ptr<VaeDecoder> &out) {
  H3_TRY(backend.prepare({"vae_embed", "vae_finish", "vae_producer_rms_f16", "vae_producer_ln_f16",
                          "vae_producer_rms_i8", "vae_producer_i8", "vae_producer_ffn_i8", "vae_qkv_i8", "vae_qkv_f16",
                          "vae_residual_i8", "vae_residual_f16", "vae_proj_f16", "vae_glu_i8", "vae_glu_f16",
                          "vae_attention"}));
  auto d = std::unique_ptr<VaeDecoder>(new VaeDecoder(backend));
  d->w_ = std::move(weights);
  H3_TRY(d->allocateActivations());
  out = std::move(d);
  return {};
}

Status VaeDecoder::allocateActivations() {
  const size_t M = static_cast<size_t>(device::kVaeBatch) * S_;
  H3_TRY(b_.allocate(M * D * 4, x_, "VAE residual"));
  H3_TRY(b_.allocate(M * D * 2, a16_, "VAE fp16 input"));
  H3_TRY(b_.allocate(M * D * 2, o_, "VAE attention output"));
  H3_TRY(b_.allocate(M * F * 2, hidden_, "VAE SwiGLU output"));
  H3_TRY(b_.allocate(w_->anyInt8() ? M * F : 16, aq_, "VAE int8 input"));
  H3_TRY(b_.allocate(M * 4, sa_, "VAE row scales"));
  const size_t heads = static_cast<size_t>(device::kVaeBatch) * H * Spad_ * HD * 2;
  for (__strong id<MTLBuffer> *t : {&q_, &k_, &v_}) {
    H3_TRY(b_.allocate(heads, *t, "VAE Q / K / V"));
    std::memset((*t).contents, 0, heads);  // pad rows must stay finite
  }
  H3_TRY(b_.allocate(256, dummy_, "VAE dummy"));
  H3_TRY(b_.allocate(16, bad_, "VAE non-finite counter"));
  return {};
}

Status VaeDecoder::ensure(size_t bytes, __strong id<MTLBuffer> &buffer, size_t &have, const char *what) {
  if (have >= bytes) return {};
  buffer = nil;
  H3_TRY(b_.allocate(bytes, buffer, what));
  have = bytes;
  return {};
}

void VaeDecoder::setProfile(bool on) {
  profile_ = on;
  stageMs_.fill(0);
  stageDispatches_.fill(0);
}

static VaeParams baseParams(int32_t rows, int32_t S, int32_t Spad) {
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

// X fp32 (norm) or fp16 (plain) -> fp16 y, or int8 + scale into the shared int8 input
void VaeDecoder::producer(Recorder &r, int stage, const char *kernel, id<MTLBuffer> x, int32_t rows, int32_t K,
                          int32_t groups, metal::BufferView normWeight, metal::BufferView normBias,
                          metal::BufferView signs, id<MTLBuffer> y) {
  auto e = r.begin(stage);
  VaeParams p = baseParams(rows, S_, Spad_);
  p.K = K;
  [e setComputePipelineState:b_.pipeline(kernel)];
  metal::bind(e, x, 0);
  metal::bind(e, aq_, 1);
  metal::bind(e, sa_, 2);
  metal::bind(e, y ? y : dummy_, 3);
  metal::bind(e, normWeight ? normWeight : metal::BufferView{dummy_, 0}, 4);
  metal::bind(e, normBias ? normBias : metal::BufferView{dummy_, 0}, 5);
  metal::bind(e, signs ? signs : metal::BufferView{dummy_, 0}, 6);
  metal::bindBytes(e, p, 7);
  metal::dispatch(e, MTLSizeMake(rows, 1, 1), MTLSizeMake(16 * (K / 128) / groups, 1, 1));
  r.end();
}

void VaeDecoder::gemm(Recorder &r, int stage, const char *kernel, int32_t tm, id<MTLBuffer> a, int32_t rows, int32_t N,
                      int32_t K, metal::BufferView w, metal::BufferView scale, metal::BufferView bias, id<MTLBuffer> y,
                      size_t yOffset, int32_t ldo, metal::BufferView aux, metal::BufferView aux2) {
  auto e = r.begin(stage);
  VaeParams p = baseParams(rows, S_, Spad_);
  p.N = N;
  p.K = K;
  p.ldo = ldo;
  [e setComputePipelineState:b_.pipeline(kernel)];
  metal::bind(e, a, 0);
  metal::bind(e, w, 1);
  metal::bind(e, sa_, 2);
  metal::bind(e, scale ? scale : metal::BufferView{dummy_, 0}, 3);
  metal::bind(e, bias, 4);
  metal::bind(e, metal::BufferView{y ? y : dummy_, yOffset}, 5);
  metal::bind(e, aux ? aux : metal::BufferView{dummy_, 0}, 6);
  metal::bindBytes(e, p, 7);
  metal::bind(e, q_, 8);
  metal::bind(e, k_, 9);
  metal::bind(e, v_, 10);
  metal::bind(e, aux2 ? aux2 : metal::BufferView{dummy_, 0}, 11);
  const int32_t tn = H3_VAE_TN;
  metal::dispatch(e, MTLSizeMake((N + tn - 1) / tn, (rows + tm - 1) / tm, 1), MTLSizeMake(32 * H3_VAE_NSG, 1, 1));
  r.end();
}

void VaeDecoder::glu(Recorder &r, bool int8, id<MTLBuffer> a, int32_t rows, const model::VaeLayerWeights &w) {
  auto e = r.begin(Fc1);
  VaeParams p = baseParams(rows, S_, Spad_);
  p.N = F;
  p.K = D;
  [e setComputePipelineState:b_.pipeline(int8 ? "vae_glu_i8" : "vae_glu_f16")];
  metal::bind(e, a, 0);
  metal::bind(e, w.fc1, 1);
  metal::bind(e, sa_, 2);
  metal::bind(e, w.fc1Scale ? w.fc1Scale : metal::BufferView{dummy_, 0}, 3);
  metal::bind(e, w.fc1Bias, 4);
  metal::bind(e, hidden_, 5);
  metal::bindBytes(e, p, 6);
  const device::Tile t = device::kVaeTile;
  metal::dispatch(e, MTLSizeMake(F / t.tn, (rows + t.tm - 1) / t.tm, 1), MTLSizeMake(32 * t.nsg, 1, 1));
  r.end();
}

void VaeDecoder::attention(Recorder &r, int32_t tiles) {
  auto e = r.begin(Attention);
  VaeParams p = baseParams(tiles * S_, S_, Spad_);
  p.ldo = D;
  const int32_t ns = device::kVaeAttentionSimdgroups;
  [e setComputePipelineState:b_.pipeline("vae_attention")];
  metal::bind(e, q_, 0);
  metal::bind(e, k_, 1);
  metal::bind(e, v_, 2);
  metal::bind(e, o_, 3);
  metal::bindBytes(e, p, 4);
  metal::dispatch(e, MTLSizeMake((Spad_ + 16 * ns - 1) / (16 * ns), H, tiles), MTLSizeMake(32 * ns, 1, 1));
  r.end();
}

void VaeDecoder::embed(Recorder &r, int32_t tiles, const int32_t *origins, int32_t latentFrames, int32_t LH,
                       int32_t LW) {
  auto e = r.begin(Embed);
  VaeParams p = baseParams(tiles * S_, S_, Spad_);
  p.LH = LH;
  p.LW = LW;
  p.T = latentFrames;
  const model::VaeGlobalWeights &g = w_->globals();
  [e setComputePipelineState:b_.pipeline("vae_embed")];
  metal::bind(e, latents_, 0);
  metal::bind(e, g.embed, 1);
  metal::bind(e, g.embedBias, 2);
  metal::bind(e, g.registers, 3);
  metal::bind(e, x_, 4);
  [e setBytes:origins length:16 * tiles atIndex:5];
  metal::bindBytes(e, p, 6);
  [e dispatchThreads:MTLSizeMake(D / 4, tiles * S_, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
  r.end();
}

// One transformer layer over the batch rows.
void VaeDecoder::layer(Recorder &r, int l, int32_t tiles) {
  const int32_t M = tiles * S_;
  const model::VaeLayerWeights &w = w_->layer(l);
  const model::VaeGlobalWeights &g = w_->globals();
  const device::Tile tq = device::kVaeQkvTile, t = device::kVaeTile;
  const int32_t groups = device::kVaeProducerGroups;
  // attention half
  if (w.int8Qkv()) producer(r, Norm, "vae_producer_rms_i8", x_, M, D, groups, w.norm1, {}, g.signs2048, nil);
  else producer(r, Norm, "vae_producer_rms_f16", x_, M, D, groups, w.norm1, {}, {}, a16_);
  gemm(r, Qkv, w.int8Qkv() ? "vae_qkv_i8" : "vae_qkv_f16", tq.tm, w.int8Qkv() ? aq_ : a16_, M, 3 * D, D, w.qkv,
       w.qkvScale, w.qkvBias, nil, 0, 0, g.cos, g.sin);
  attention(r, tiles);
  if (w.int8Out()) producer(r, Quantize, "vae_producer_i8", o_, M, D, groups, {}, {}, g.signs2048, nil);
  gemm(r, Out, w.int8Out() ? "vae_residual_i8" : "vae_residual_f16", t.tm, w.int8Out() ? aq_ : o_, M, D, D, w.out,
       w.outScale, w.outBias, x_, 0, D, w.scale1, {});
  // MLP half
  if (w.int8Fc1()) producer(r, Norm, "vae_producer_rms_i8", x_, M, D, groups, w.norm2, {}, g.signs2048, nil);
  else producer(r, Norm, "vae_producer_rms_f16", x_, M, D, groups, w.norm2, {}, {}, a16_);
  glu(r, w.int8Fc1(), w.int8Fc1() ? aq_ : a16_, M, w);
  if (w.int8Fc2())
    producer(r, Quantize, "vae_producer_ffn_i8", hidden_, M, F, device::kVaeFfnProducerGroups, {}, {}, g.signs8192,
             nil);
  gemm(r, Fc2, w.int8Fc2() ? "vae_residual_i8" : "vae_residual_f16", t.tm, w.int8Fc2() ? aq_ : hidden_, M, D, F, w.fc2,
       w.fc2Scale, w.fc2Bias, x_, 0, D, w.scale2, {});
}

void VaeDecoder::head(Recorder &r, int32_t tiles, size_t outRow0) {
  const int32_t M = tiles * S_;
  const model::VaeGlobalWeights &g = w_->globals();
  producer(r, LayerNorm, "vae_producer_ln_f16", x_, M, D, device::kVaeProducerGroups, g.lnWeight, g.lnBias, {}, a16_);
  gemm(r, Proj, "vae_proj_f16", device::kVaeTile.tm, a16_, M, NP, D, g.proj, {}, g.projBias, out_,
       outRow0 * NP * sizeof(float), NP, {}, {});
}

Status VaeDecoder::decode(const float *latents, int32_t latentFrames, const model::VideoGeometry &geometry,
                          const model::VaePlan &plan, uint8_t *frames, DecodeStats &stats) {
  const double tStart = now();
  stats = DecodeStats{};
  const int32_t LH = geometry.latentHeight, LW = geometry.latentWidth;
  const int32_t HP = LH * 16, WP = LW * 16;
  const int32_t nty = static_cast<int32_t>(plan.ys.size()), ntx = static_cast<int32_t>(plan.xs.size());
  const int32_t tiles = nty * ntx, CF = 4 * plan.ct, NF0 = CF - plan.fpp, fov = plan.fov;
  const int64_t outFrames = plan.outputFrames();
  if (nty > H3_VAE_MAX_TILES || ntx > H3_VAE_MAX_TILES || fov > NF0 || fov != 4 * plan.tov - plan.fpp)
    return error(Code::Argument, "unsupported VAE decode plan");
  H3_TRY(ensure(static_cast<size_t>(tiles) * S_ * NP * sizeof(float), out_, outBytes_, "VAE proj_out tokens"));
  const size_t fsz = static_cast<size_t>(HP) * WP * 3;
  if (overlapBytes_ < fsz * fov * 4) {
    for (int i = 0; i < 2; ++i) H3_TRY(b_.allocate(fsz * fov * 4, overlap_[i], "VAE overlap frames"));
    overlapBytes_ = fsz * fov * 4;
  }
  if (stageBytes_ < fsz * NF0) {
    for (int i = 0; i < 2; ++i) H3_TRY(b_.allocate(fsz * NF0, stage_[i], "VAE frame staging"));
    stageBytes_ = fsz * NF0;
  }
  const size_t latentBytes = static_cast<size_t>(latentFrames) * LH * LW * H3_VAE_LATENT * 4;
  H3_TRY(ensure(latentBytes, latents_, latentBytes_, "VAE latents"));
  std::memcpy(latents_.contents, latents, latentBytes);
  stats.uploadS = now() - tStart;
  std::memset(bad_.contents, 0, 16);
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
  std::vector<id<MTLCommandBuffer>> all;
  id<MTLCommandBuffer> previousFinish = nil;
  double copyS = 0;
  // frames [first, first + count) of a staging buffer, clipped to the output (pad frames are cut)
  auto copyFrames = [&](id<MTLCommandBuffer> cb, int slot, int64_t first, int64_t count) {
    [cb waitUntilCompleted];
    const double t0 = now();
    const int64_t n = std::min<int64_t>(count, outFrames - first);
    if (n > 0) std::memcpy(frames + first * fsz, stage_[slot].contents, fsz * n);
    copyS += now() - t0;
  };
  for (int32_t c = 0; c < plan.chunks; ++c) {
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
      Recorder r(*this);
      embed(r, nb, origins.data(), latentFrames, LH, LW);
      for (int l = 0; l < H3_VAE_LAYERS; ++l) layer(r, l, nb);
      head(r, nb, static_cast<size_t>(k0) * S_);
      if (id<MTLCommandBuffer> cb = r.commit()) all.push_back(cb);
    }
    Recorder r(*this);
    auto e = r.begin(Finish);
    fp.has_prev = c > 0;
    fp.mode = 0;
    [e setComputePipelineState:b_.pipeline("vae_finish")];
    metal::bind(e, out_, 0);
    metal::bind(e, overlap_[(c + 1) % 2], 1);
    metal::bind(e, overlap_[c % 2], 2);
    metal::bind(e, stage_[c % 2], 3);
    metal::bind(e, bad_, 4);
    metal::bindBytes(e, fp, 5);
    [e dispatchThreads:MTLSizeMake(WP, HP, NF0 + fov) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    r.end();
    id<MTLCommandBuffer> finish = r.commit();
    if (!finish) {  // profile: the finish ran synchronously; a completed placeholder for the copy
      finish = [b_.queue() commandBuffer];
      [finish commit];
    }
    all.push_back(finish);
    if (previousFinish) copyFrames(previousFinish, (c - 1) % 2, int64_t(c - 1) * NF0, NF0);
    previousFinish = finish;
  }
  copyFrames(previousFinish, (plan.chunks - 1) % 2, int64_t(plan.chunks - 1) * NF0, NF0);
  {  // tail: the last clip's overlap frames
    Recorder r(*this);
    auto e = r.begin(Finish);
    fp.mode = 1;
    [e setComputePipelineState:b_.pipeline("vae_finish")];
    metal::bind(e, out_, 0);
    metal::bind(e, overlap_[(plan.chunks - 1) % 2], 1);
    metal::bind(e, dummy_, 2);
    metal::bind(e, stage_[0], 3);
    metal::bind(e, bad_, 4);
    metal::bindBytes(e, fp, 5);
    [e dispatchThreads:MTLSizeMake(WP, HP, fov) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    r.end();
    id<MTLCommandBuffer> finish = r.commit();
    if (!finish) {
      finish = [b_.queue() commandBuffer];
      [finish commit];
    }
    all.push_back(finish);
    copyFrames(finish, 0, int64_t(plan.chunks) * NF0, fov);
  }
  double gpu = 0, g0 = 1e30, g1 = 0;
  for (id<MTLCommandBuffer> cb : all) {
    if (cb.error) return error(Code::Gpu, "VAE command buffer: %s", cb.error.localizedDescription.UTF8String);
    if (cb.GPUEndTime > 0) {
      gpu += cb.GPUEndTime - cb.GPUStartTime;
      g0 = std::min(g0, static_cast<double>(cb.GPUStartTime));
      g1 = std::max(g1, static_cast<double>(cb.GPUEndTime));
    }
  }
  stats.wallS = now() - tStart;
  stats.gpuS = gpu;
  stats.hostCopyS = copyS;
  stats.gpuSpanS = g1 - g0;
  stats.nonfinite = static_cast<const uint32_t *>(bad_.contents)[0];
  stats.deviceMb = b_.allocatedMb();
  return {};
}

} // namespace h3::ops
