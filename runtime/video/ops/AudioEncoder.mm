#include "video/ops/AudioEncoder.hpp"

#include "metal/abi/video/AudioABI.h"

#include "AwakeClock.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace splash::video::ops {
namespace {

constexpr int32_t kP = H3_AUDIO_PAD, kSlack = H3_AUDIO_SLACK, kTail = H3_AUDIO_ENC_TAIL, kTileM = 64;

int32_t domainRows(int32_t T) { return (2 * (T + 2 * kP) + kTileM - 1) / kTileM * kTileM; }
size_t pairElems(int32_t T, int32_t C) { return size_t(2 * kSlack + kTail + domainRows(T)) * C; }

struct TileShape {
  const char *name;
  int32_t tm, tn, nsg;
};
TileShape tileFor(int32_t cout) {
  if (cout >= 128) return {"64x128x4", 64, 128, 4};
  if (cout == 64) return {"64x64x2", 64, 64, 2};
  if (cout == 32) return {"64x32x1", 64, 32, 1};
  return {"64x16x1", 64, 16, 1};
}

// Host fp32 helpers (pre_block): out[M][N] = A[M][K] Bt[K][N] (+ bias), blocks of 64 rows x 64 columns in parallel,
// each output a sequential fp32 sum over k.
void gemm(const float *A, const float *Bt, const float *bias, int32_t M, int32_t K, int32_t N, float *out) {
  constexpr int32_t RB = 64, CB = 64;
  const int32_t rb = (M + RB - 1) / RB, cb = (N + CB - 1) / CB;
  dispatch_apply(size_t(rb) * cb, DISPATCH_APPLY_AUTO, ^(size_t task) {
    const int32_t r0 = int32_t(task / cb) * RB, c0 = int32_t(task % cb) * CB;
    const int32_t rows = std::min(RB, M - r0), cols = std::min(CB, N - c0);
    float acc[RB][CB];
    for (int32_t r = 0; r < rows; ++r)
      for (int32_t c = 0; c < CB; ++c) acc[r][c] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      const float *b = Bt + size_t(k) * N + c0;
      if (cols == CB) {
        for (int32_t r = 0; r < rows; ++r) {
          const float a = A[size_t(r0 + r) * K + k];
          for (int32_t c = 0; c < CB; ++c) acc[r][c] += a * b[c];
        }
      } else {
        for (int32_t r = 0; r < rows; ++r) {
          const float a = A[size_t(r0 + r) * K + k];
          for (int32_t c = 0; c < cols; ++c) acc[r][c] += a * b[c];
        }
      }
    }
    for (int32_t r = 0; r < rows; ++r)
      for (int32_t c = 0; c < cols; ++c) out[size_t(r0 + r) * N + c0 + c] = acc[r][c] + (bias ? bias[c0 + c] : 0.0f);
  });
}

void layerNorm(const float *x, const float *w, const float *b, int32_t T, int32_t C, float *y) {
  for (int32_t t = 0; t < T; ++t) {
    const float *row = x + size_t(t) * C;
    double s = 0, s2 = 0;
    for (int32_t c = 0; c < C; ++c) s += row[c];
    const double mean = s / C;
    for (int32_t c = 0; c < C; ++c) s2 += (row[c] - mean) * (row[c] - mean);
    const float m = float(mean), inv = float(1.0 / std::sqrt(s2 / C + 1e-5));
    for (int32_t c = 0; c < C; ++c) y[size_t(t) * C + c] = (row[c] - m) * inv * w[c] + b[c];
  }
}

float geluTanh(float x) {
  return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

} // namespace

AudioEncoder::AudioEncoder(metal::MetalBackend &backend, AudioEncoderWeights weights)
    : b_(backend), w_(std::move(weights)), null_(allocate(backend, 16, "audio encoder null")) {}

void AudioEncoder::reserve(int32_t T0, int64_t samples) {
  if (samples > wavCapacity_) {
    wav_ = allocate(b_, size_t(2) * samples * 4, "audio encoder waveform");
    wavCapacity_ = samples;
  }
  if (T0 <= capacity_) return;
  size_t f32 = 0, pair = 0;
  int32_t T = T0, C = kEncoderDim;
  for (int i = 0; i < kBlocks; ++i) {
    f32 = std::max(f32, size_t(2) * T * C);
    pair = std::max(pair, pairElems(T, C));
    T /= kRates[i];
    C *= 2;
  }
  f32 = std::max(f32, size_t(2) * T * kLatentDim);
  pair = std::max(pair, pairElems(T, kLatentDim));
  a_.hi = allocate(b_, pair * 2, "audio encoder activations");
  a_.lo = allocate(b_, pair * 2, "audio encoder activations");
  std::memset(a_.hi.contents(), 0, a_.hi.sizeBytes());
  std::memset(a_.lo.contents(), 0, a_.lo.sizeBytes());
  x_ = allocate(b_, f32 * 4, "audio encoder residual");
  r_ = allocate(b_, f32 * 4, "audio encoder conv output");
  capacity_ = T0;
}

void AudioEncoder::conv(Encoder &e, const AudioConv &w, int32_t T, int32_t dil, int32_t pad, bool residual) {
  AudioConvParams p{};
  p.Cin = w.cin;
  p.Cout = w.cout;
  p.T = T;
  p.Mrows = domainRows(T);
  p.taps = w.taps;
  p.dil = dil;
  p.pad = pad;
  p.stride = 1;
  p.Tout = T;
  const TileShape t = tileFor(w.cout);
  e.pipeline(std::string(residual ? "audio_conv_res_" : "audio_conv_f32_") + t.name);
  const size_t inOff = size_t(kSlack) * w.cin * 2;
  e.bind({a_.hi, inOff}, 0);
  e.bind({a_.lo, inOff}, 1);
  e.bind(w.w, 2);
  e.bind(w.wl, 3);
  e.bind(w.b, 4);
  e.bind(residual ? x_ : r_, 5);
  e.bind(residual ? x_ : null_, 6);
  e.bind(null_, 7);
  e.bind(null_, 8);
  e.bytes(p, 9);
  e.dispatch(size3(w.cout / t.tn, p.Mrows / t.tm, 1), size3(32 * t.nsg, 1, 1));
}

void AudioEncoder::down(Encoder &e, const AudioConv &w, int32_t T, int32_t stride) {
  AudioConvParams p{};
  p.Cin = w.cin / stride;
  p.Cout = w.cout;
  p.T = T;
  p.Tout = T / stride;
  p.Mrows = (p.Tout + kTileM - 1) / kTileM * kTileM;
  p.taps = w.taps;
  p.dil = 1;
  p.pad = (stride + 1) / 2;
  p.stride = stride;
  e.pipeline("audio_down_64x128x4");
  const size_t inOff = size_t(kSlack) * p.Cin * 2;
  e.bind({a_.hi, inOff}, 0);
  e.bind({a_.lo, inOff}, 1);
  e.bind(w.w, 2);
  e.bind(w.wl, 3);
  e.bind(w.b, 4);
  e.bind(x_, 5);
  e.bytes(p, 9);
  e.dispatch(size3(w.cout / 128, p.Mrows / 64, 2), size3(128, 1, 1));
}

void AudioEncoder::snake(Encoder &e, const metal::MetalBuffer &x, int32_t T, int32_t C, BufferView act) {
  const AudioActParams p{C, T, 0, 0};
  e.pipeline("audio_snake");
  e.bind(x, 0);
  const size_t off = size_t(kSlack) * C * 2;
  e.bind({a_.hi, off}, 1);
  e.bind({a_.lo, off}, 2);
  e.bind(act, 3);
  e.bytes(p, 4);
  e.dispatchThreads(size3(C, 2 * (T + 2 * kP), 1), size3(64, 4, 1));
}

void AudioEncoder::record(Encoder &e, int32_t T0, int64_t samples) {
  const AudioEncoderWeights &w = w_;
  {
    const AudioActParams p{kEncoderDim, T0, int32_t(samples), 0};
    e.pipeline("audio_enc_in");
    e.bind(wav_, 0);
    e.bind(w.convIn.w, 1);
    e.bind(w.convIn.wl, 2);
    e.bind(w.convIn.b, 3);
    e.bind(x_, 4);
    e.bytes(p, 5);
    e.dispatchThreads(size3(kEncoderDim, 2 * T0, 1), size3(64, 4, 1));
  }
  int32_t T = T0, C = kEncoderDim;
  for (int i = 0; i < kBlocks; ++i) {
    const AudioEncoderWeights::Block &blk = w.blocks[i];
    for (int u = 0; u < 3; ++u) {
      const AudioEncoderWeights::Unit &unit = blk.units[u];
      const int32_t d = kDilations[u];
      snake(e, x_, T, C, unit.act1);
      conv(e, unit.conv7, T, d, 3 * d, false);
      snake(e, r_, T, C, unit.act2);
      conv(e, unit.conv1, T, 1, 0, true);
    }
    snake(e, x_, T, C, blk.act);
    down(e, blk.down, T, kRates[i]);
    T /= kRates[i];
    C *= 2;
  }
  snake(e, x_, T, C, w.finalAct);
  conv(e, w.convOut, T, 1, 1, false);
}

void AudioEncoder::preBlock(const float *x, int32_t T, float *out) const {
  const AudioEncoderWeights &w = w_;
  constexpr int32_t D = kLatentDim, Hd = kLatentDim / kHeads, L = kLatent;
  std::vector<float> h1(size_t(T) * D), qkv(size_t(T) * 3 * D), o(size_t(T) * kHeads * Hd);
  layerNorm(x, w.norm1W, w.norm1B, T, D, h1.data());
  gemm(h1.data(), w.qkvT, nullptr, T, D, 3 * D, qkv.data());
  for (int32_t t = 0; t < T; ++t) {
    float *row = qkv.data() + size_t(t) * 3 * D;
    for (int32_t c = 0; c < D; ++c) {
      row[c] += w.qBias[c];
      row[2 * D + c] += w.vBias[c];
    }
  }
  // causal attention, scale 1 / sqrt(256), one task per (query row, head)
  const float scale = 1.0f / std::sqrt(float(Hd));
  const float *Q = qkv.data();
  float *O = o.data();
  dispatch_apply(size_t(T) * kHeads, DISPATCH_APPLY_AUTO, ^(size_t task) {
    const int32_t t = int32_t(task / kHeads), h = int32_t(task % kHeads);
    const float *q = Q + size_t(t) * 3 * D + h * Hd;
    std::vector<float> s(size_t(t) + 1);
    float mx = -INFINITY;
    for (int32_t j = 0; j <= t; ++j) {
      const float *k = Q + size_t(j) * 3 * D + D + h * Hd;
      float acc = 0.0f;
      for (int32_t d = 0; d < Hd; ++d) acc += q[d] * k[d];
      s[j] = acc * scale;
      mx = std::max(mx, s[j]);
    }
    float sum = 0.0f;
    for (int32_t j = 0; j <= t; ++j) {
      s[j] = std::exp(s[j] - mx);
      sum += s[j];
    }
    float *dst = O + (size_t(t) * kHeads + h) * Hd;
    for (int32_t d = 0; d < Hd; ++d) dst[d] = 0.0f;
    for (int32_t j = 0; j <= t; ++j) {
      const float p = s[j] / sum;
      const float *v = Q + size_t(j) * 3 * D + 2 * D + h * Hd;
      for (int32_t d = 0; d < Hd; ++d) dst[d] += p * v[d];
    }
  });
  // mean over heads, adaptive average pool 256 -> 32 (groups of 8), attn.proj
  std::vector<float> pooled(size_t(T) * L), attn(size_t(T) * L), h(size_t(T) * L);
  constexpr int32_t G = Hd / L;
  for (int32_t t = 0; t < T; ++t)
    for (int32_t g = 0; g < L; ++g) {
      float acc = 0.0f;
      for (int32_t d = g * G; d < (g + 1) * G; ++d) {
        float m = 0.0f;
        for (int32_t hh = 0; hh < kHeads; ++hh) m += o[(size_t(t) * kHeads + hh) * Hd + d];
        acc += m / float(kHeads);
      }
      pooled[size_t(t) * L + g] = acc / float(G);
    }
  gemm(pooled.data(), w.attnProj.wt, w.attnProj.b, T, L, L, attn.data());
  layerNorm(x, w.norm3W, w.norm3B, T, D, h1.data());
  gemm(h1.data(), w.proj.wt, w.proj.b, T, D, L, h.data());
  for (size_t i = 0; i < h.size(); ++i) h[i] += attn[i];
  // h + mlp(norm2(h)); the MLP normalises again (mlp.norm)
  std::vector<float> z(size_t(T) * L), z2(size_t(T) * L), g0(size_t(T) * 2 * L), g1(size_t(T) * 2 * L), m(size_t(T) * L);
  layerNorm(h.data(), w.norm2W, w.norm2B, T, L, z.data());
  layerNorm(z.data(), w.mlpNormW, w.mlpNormB, T, L, z2.data());
  gemm(z2.data(), w.w0.wt, w.w0.b, T, L, 2 * L, g0.data());
  gemm(z2.data(), w.w1.wt, w.w1.b, T, L, 2 * L, g1.data());
  for (size_t i = 0; i < g0.size(); ++i) g0[i] = geluTanh(g0[i]) * g1[i];
  gemm(g0.data(), w.w2.wt, w.w2.b, T, 2 * L, L, m.data());
  for (size_t i = 0; i < h.size(); ++i) h[i] += m[i];
  gemm(h.data(), w.meanProj.wt, w.meanProj.b, T, L, L, m.data());
  for (int32_t t = 0; t < T; ++t)
    for (int32_t c = 0; c < L; ++c) out[size_t(t) * L + c] = (m[size_t(t) * L + c] - w.mean[c]) / w.std[c];
}

void AudioEncoder::encode(const float *wav, int64_t samples, std::vector<float> &latents, AudioEncodeStats &stats) {
  const auto t0 = splash::AwakeClock::now();
  stats = AudioEncodeStats{};
  if (samples < 1) throw std::invalid_argument("audio encoder: no samples");
  const int64_t T = (samples + kHop - 1) / kHop;
  if (T * kHop > INT32_MAX / 4) throw std::invalid_argument("audio encoder: soundtrack too long");
  const int32_t T0 = int32_t(T * kHop);
  reserve(T0, samples);
  std::memcpy(wav_.contents(), wav, size_t(2) * samples * 4);
  Encoder e(b_);
  record(e, T0, samples);
  stats.gpuS = run(b_, e);
  const auto *x = static_cast<const float *>(r_.contents());
  std::vector<float> ch(size_t(2) * T * kLatent);
  for (int b = 0; b < 2; ++b) preBlock(x + size_t(b) * T * kLatentDim, int32_t(T), ch.data() + size_t(b) * T * kLatent);
  latents.resize(size_t(T) * 2 * kLatent);
  for (int64_t t = 0; t < T; ++t)
    for (int b = 0; b < 2; ++b)
      std::memcpy(&latents[(size_t(t) * 2 + b) * kLatent], &ch[(size_t(b) * T + t) * kLatent], kLatent * 4);
  for (float v : latents) stats.nonfinite += !std::isfinite(v);
  stats.wallS = splash::millisecondsSince(t0) / 1e3;
}

} // namespace splash::video::ops
