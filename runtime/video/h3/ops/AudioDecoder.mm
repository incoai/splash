#include "video/h3/ops/AudioDecoder.hpp"

#include "metal/abi/video/AudioABI.h"

#include "AwakeClock.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace h3::ops {
namespace {

constexpr int32_t kP = H3_AUDIO_PAD, kSlack = H3_AUDIO_SLACK, kTileM = 64, kSimtThreads = 256;

int32_t domainRows(int32_t T) { return (2 * (T + 2 * kP) + kTileM - 1) / kTileM * kTileM; }
size_t pairElems(int32_t T, int32_t C) { return size_t(2 * kSlack + domainRows(T)) * C; }

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

} // namespace

struct AudioDecoder::Conv {
  enum Epi { F32, RES, HALF2, WAV } epi = F32;
  bool ups = false;
  const Pair *in = nullptr;
  int32_t T = 0;  // input rows per channel
  const model::AudioConv *w = nullptr;
  int32_t dil = 1, pad = 0, stride = 1, upad = 0, Tout = 0;
  metal::MetalBuffer out, res;
  const Pair *outPair = nullptr;
};

AudioDecoder::AudioDecoder(metal::MetalBackend &backend, std::unique_ptr<model::AudioWeights> weights)
    : b_(backend), w_(std::move(weights)), null_(allocate(backend, 16, "audio null")) {}

void AudioDecoder::reserve(int32_t n) {
  if (n <= capacity_) return;
  size_t f32 = 0, pairA = std::max(pairElems(n, H3_AUDIO_LATENT), pairElems(n, model::AudioWeights::kDecoderDim));
  int32_t T = n;
  for (int s = 0; s < model::AudioWeights::kStages; ++s) {
    T *= model::AudioWeights::kRates[s];
    const int32_t C = model::AudioWeights::kDecoderDim >> (s + 1);
    f32 = std::max(f32, size_t(2) * T * C);
    pairA = std::max(pairA, pairElems(T, C));
  }
  const size_t pairB = pairElems(n, model::AudioWeights::kLatentDim);
  a_.hi = allocate(b_, pairA * 2, "audio activations");
  a_.lo = allocate(b_, pairA * 2, "audio activations");
  b2_.hi = allocate(b_, pairB * 2, "audio activations");
  b2_.lo = allocate(b_, pairB * 2, "audio activations");
  for (auto &blk : blocks_) blk = allocate(b_, f32 * 4, "audio residual");
  x0_ = allocate(b_, f32 * 4, "audio residual");
  r_ = allocate(b_, f32 * 4, "audio conv output");
  rows_ = allocate(b_, size_t(2) * n * H3_AUDIO_LATENT * 4, "audio latents");
  wav_ = allocate(b_, size_t(2) * samples(n) * 4, "audio waveform");
  for (const Pair *p : {&a_, &b2_}) {
    std::memset(p->hi.contents(), 0, p->hi.sizeBytes());
    std::memset(p->lo.contents(), 0, p->lo.sizeBytes());
  }
  capacity_ = n;
}

void AudioDecoder::conv(Encoder &e, const Conv &c) {
  const model::AudioConv &w = *c.w;
  AudioConvParams p{};
  p.Cin = w.cin;
  p.Cout = w.cout;
  p.T = c.T;
  p.Mrows = domainRows(c.T);
  p.taps = w.taps;
  p.dil = c.dil;
  p.pad = c.pad;
  p.stride = c.stride;
  p.upad = c.upad;
  p.Tout = c.Tout;
  const bool simt = w.cout < 16 || w.cin < 16;
  std::string name;
  TileShape t{};
  if (simt) {
    if (c.epi == Conv::WAV) name = "audio_small_wav_8";
    else if (c.ups) name = "audio_small_ups_16x8";
    else name = c.epi == Conv::RES ? "audio_small_res_8" : "audio_small_f32_8";
  } else {
    t = tileFor(w.cout);
    const char *k = c.ups ? "audio_ups_f32_"
                          : (c.epi == Conv::RES ? "audio_conv_res_" : (c.epi == Conv::HALF2 ? "audio_conv_half_" : "audio_conv_f32_"));
    name = std::string(k) + t.name;
  }
  e.pipeline(name);
  const size_t inOff = size_t(kSlack) * w.cin * 2;
  e.bind({c.in->hi, inOff}, 0);
  e.bind({c.in->lo, inOff}, 1);
  e.bind(w.w, 2);
  e.bind(w.wl, 3);
  e.bind(w.b ? w.b : BufferView{null_}, 4);
  e.bind(c.out ? c.out : null_, 5);
  e.bind(c.res ? c.res : null_, 6);
  const size_t outOff = size_t(kSlack) * w.cout * 2;
  if (c.outPair) {
    e.bind({c.outPair->hi, outOff}, 7);
    e.bind({c.outPair->lo, outOff}, 8);
  } else {
    e.bind(null_, 7);
    e.bind(null_, 8);
  }
  e.bytes(p, 9);
  const int32_t phases = c.ups ? c.stride : 1;
  if (simt)
    e.dispatch(size3((p.Mrows + kSimtThreads - 1) / kSimtThreads, 1, phases), size3(kSimtThreads, 1, 1));
  else
    e.dispatch(size3(w.cout / t.tn, p.Mrows / t.tm, phases), size3(32 * t.nsg, 1, 1));
}

void AudioDecoder::act(Encoder &e, const metal::MetalBuffer &x, int32_t T, int32_t C, BufferView snake) {
  const int32_t ct = std::min(C, 32);
  const AudioActParams p{C, T, 0, 0};
  e.pipeline("audio_act_" + std::to_string(ct));
  e.bind(x, 0);
  const size_t off = size_t(kSlack) * C * 2;
  e.bind({a_.hi, off}, 1);
  e.bind({a_.lo, off}, 2);
  e.bind(snake, 3);
  e.bind(w_->filter, 4);
  e.bytes(p, 5);
  e.dispatch(size3(C / ct, (T + H3_AUDIO_ACT_ROWS - 1) / H3_AUDIO_ACT_ROWS, 2), size3(256, 1, 1));
}

void AudioDecoder::zeroPads(Encoder &e, const Pair &pair, int32_t T, int32_t C) {
  const AudioActParams p{C, T, 0, 0};
  e.pipeline("audio_zero_pads");
  const size_t off = size_t(kSlack) * C * 2;
  e.bind({pair.hi, off}, 0);
  e.bind({pair.lo, off}, 1);
  e.bytes(p, 2);
  e.dispatchThreads(size3(C, 4 * kP, 1), size3(std::min(C, 256), 1, 1));
}

void AudioDecoder::encode(Encoder &e, int32_t n) {
  const model::AudioWeights &w = *w_;
  {  // latents -> hi / lo padded [2][n][32]
    const AudioActParams p{H3_AUDIO_LATENT, n, 0, 0};
    e.pipeline("audio_input");
    e.bind(rows_, 0);
    e.bind(w.mean, 1);
    e.bind(w.std, 2);
    const size_t off = size_t(kSlack) * H3_AUDIO_LATENT * 2;
    e.bind({a_.hi, off}, 3);
    e.bind({a_.lo, off}, 4);
    e.bytes(p, 5);
    e.dispatchThreads(size3(H3_AUDIO_LATENT, 2 * (n + 2 * kP), 1), size3(32, 8, 1));
  }
  zeroPads(e, b2_, n, model::AudioWeights::kLatentDim);
  Conv c;
  c.epi = Conv::HALF2;
  c.in = &a_;
  c.T = n;
  c.w = &w.decIn;
  c.Tout = n;
  c.outPair = &b2_;
  conv(e, c);
  zeroPads(e, a_, n, model::AudioWeights::kDecoderDim);
  c = Conv{};
  c.epi = Conv::HALF2;
  c.in = &b2_;
  c.T = n;
  c.w = &w.convPre;
  c.pad = 3;
  c.Tout = n;
  c.outPair = &a_;
  conv(e, c);
  int32_t T = n;
  for (int s = 0; s < model::AudioWeights::kStages; ++s) {
    const model::AudioStageWeights &g = w.stages[s];
    const int32_t rate = model::AudioWeights::kRates[s], C = g.up.cout, Tin = T;
    T = Tin * rate;
    c = Conv{};
    c.ups = true;
    c.in = &a_;
    c.T = Tin;
    c.w = &g.up;
    c.stride = rate;
    c.upad = (g.up.taps - rate) / 2;
    c.Tout = T;
    c.out = x0_;
    conv(e, c);
    for (int j = 0; j < 3; ++j) {
      const int32_t k = model::AudioWeights::kResKernels[j];
      for (int d = 0; d < 3; ++d) {
        const int32_t D = model::AudioWeights::kDilations[d];
        metal::MetalBuffer src = d == 0 ? x0_ : blocks_[j];
        act(e, src, T, C, g.acts[j][2 * d]);
        c = Conv{};
        c.in = &a_;
        c.T = T;
        c.w = &g.convs1[j][d];
        c.dil = D;
        c.pad = D * (k - 1) / 2;
        c.Tout = T;
        c.out = r_;
        conv(e, c);
        act(e, r_, T, C, g.acts[j][2 * d + 1]);
        c = Conv{};
        c.epi = Conv::RES;
        c.in = &a_;
        c.T = T;
        c.w = &g.convs2[j][d];
        c.pad = (k - 1) / 2;
        c.Tout = T;
        c.out = blocks_[j];
        c.res = src;
        conv(e, c);
      }
    }
    const bool last = s + 1 == model::AudioWeights::kStages;
    const AudioActParams p{C, T, 0, 0};
    e.pipeline(last ? "audio_sum3_f32" : "audio_sum3_half");
    e.bind(blocks_[0], 0);
    e.bind(blocks_[1], 1);
    e.bind(blocks_[2], 2);
    e.bind(x0_, 3);
    const size_t off = size_t(kSlack) * C * 2;
    e.bind({a_.hi, off}, 4);
    e.bind({a_.lo, off}, 5);
    e.bytes(p, 6);
    const int32_t rows = last ? 2 * T : 2 * (T + 2 * kP);
    e.dispatchThreads(size3(C, rows, 1), size3(std::min(C, 64), 256 / std::min(C, 64), 1));
  }
  act(e, x0_, T, w.convPost.cin, w.postAct);
  c = Conv{};
  c.epi = Conv::WAV;
  c.in = &a_;
  c.T = T;
  c.w = &w.convPost;
  c.pad = 3;
  c.Tout = T;
  c.out = wav_;
  conv(e, c);
}

void AudioDecoder::decode(const float *rows, int32_t n, float *wav, AudioStats &stats) {
  const auto t0 = splash::AwakeClock::now();
  stats = AudioStats{};
  if (n < 1) throw std::invalid_argument("audio: no latents");
  reserve(n);
  std::memcpy(rows_.contents(), rows, size_t(2) * n * H3_AUDIO_LATENT * 4);
  Encoder e(b_);
  encode(e, n);
  stats.gpuS = run(b_, e);
  const size_t count = size_t(2) * samples(n);
  std::memcpy(wav, wav_.contents(), count * 4);
  for (size_t i = 0; i < count; ++i) stats.nonfinite += !std::isfinite(wav[i]);
  stats.samples = samples(n);
  stats.deviceMb = deviceMb(b_);
  stats.wallS = splash::millisecondsSince(t0) / 1e3;
}

} // namespace h3::ops
