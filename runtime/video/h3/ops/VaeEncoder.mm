#include "video/h3/ops/VaeEncoder.hpp"

#include "AwakeClock.hpp"
#include "metal/abi/video/VaeEncoderABI.h"
#include "model/WeightStore.hpp"
#include "video/h3/engine/DeviceProfile.hpp"
#include "video/h3/model/WeightFiles.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace h3::ops {
namespace {

constexpr int32_t kLevels = 6;
constexpr int32_t kCh[kLevels] = {128, 256, 256, 512, 512, 1024};
constexpr int32_t kSpace[kLevels] = {2, 2, 2, 2, 1, 1};
constexpr int32_t kTime[kLevels] = {1, 2, 2, 1, 1, 1};
constexpr float kEps = 1e-6f;   // norm_eps
constexpr int32_t kTile = 256;  // tile_sample_min_height / width
constexpr int32_t kOverlap = 64;
constexpr int32_t kRatio = 16;  // spatial compression
constexpr int32_t kSlackPixels = 256;  // garbage-row reads past the last frame

std::string tileName(device::Tile t) {
  return std::to_string(t.tm) + "x" + std::to_string(t.tn) + "x" + std::to_string(t.nsg);
}

int32_t frameStride(int32_t H, int32_t W) { return ((H + 2) * (W + 2) + 15) & ~15; }

// One stitched latent tile [h][w][48].
struct Tile {
  int32_t h = 0, w = 0;
  std::vector<float> d;
  [[nodiscard]] float at(int32_t y, int32_t x, int32_t c) const { return d[(size_t(y) * w + x) * 48 + c]; }
};

// VideoVAE._blend along axis 0 (rows) or 1 (columns): b's first `extent` lines become a_tail * (1 - p / e) + b * p / e
Tile blend(const Tile &a, const Tile &b, int32_t extent, int axis) {
  const int32_t la = axis == 0 ? a.h : a.w, lb = axis == 0 ? b.h : b.w;
  extent = std::min({la, lb, extent});
  Tile out = b;
  if (extent <= 0) return out;
  for (int32_t p = 0; p < extent; ++p) {
    const float pos = static_cast<float>(p) / static_cast<float>(extent);
    const float wa = 1.0f - pos, wb = pos;
    const int32_t lines = axis == 0 ? b.w : b.h;
    for (int32_t q = 0; q < lines; ++q)
      for (int32_t c = 0; c < 48; ++c) {
        const float av = axis == 0 ? a.at(la - extent + p, q, c) : a.at(q, la - extent + p, c);
        float &bv = axis == 0 ? out.d[(size_t(p) * b.w + q) * 48 + c] : out.d[(size_t(q) * b.w + p) * 48 + c];
        const float x = av * wa;
        const float y = bv * wb;
        bv = x + y;
      }
  }
  return out;
}

} // namespace

void VaeEncoder::splitTiles(int32_t length, int32_t tile, int32_t minOverlap, std::vector<int32_t> &starts,
                            std::vector<int32_t> &lengths, std::vector<int32_t> &overlaps) {
  starts.clear();
  lengths.clear();
  overlaps.clear();
  if (tile >= length) {
    starts = {0};
    lengths = {length};
    return;
  }
  int32_t n = (length + tile - 1) / tile;
  while (tile * n - minOverlap * (n - 1) - length < 0) ++n;
  overlaps.assign(n - 1, minOverlap);
  int32_t sum = 0;
  for (int32_t o : overlaps) sum += o;
  const int32_t remaining = tile * n - sum - length;
  for (int32_t i = 0; i < remaining / kRatio; ++i) overlaps[i % (n - 1)] += kRatio;
  starts = {0};
  for (int32_t i = 0; i < n - 1; ++i) starts.push_back(starts.back() + tile - overlaps[i]);
  lengths.assign(n, tile);
}

VaeEncoder::VaeEncoder(metal::MetalBackend &backend, const std::string &vaeDir) : b_(backend) {
  loadWeights(vaeDir);
  const int32_t B = device::kVaeEncodeBatch;
  capH_ = capW_ = kTile;
  const size_t act = size_t(B) * kTile * kTile * kCh[0];  // largest fp32 activation (level 0)
  for (Act *a : {&x_[0], &x_[1], &h_, &s_}) a->buf = allocate(b_, act * 4, "VAE encoder activation");
  pixels_.buf = allocate(b_, size_t(B) * kTile * kTile * 3 * 4, "VAE encoder pixels");
  const size_t lat = size_t(B) * (kTile / kRatio) * (kTile / kRatio) * H3_VAE_ENC_MOMENTS;
  out_.buf = allocate(b_, lat * 4, "VAE encoder conv_out");
  moments_.buf = allocate(b_, lat * 4, "VAE encoder moments");
  const size_t pad = size_t(B) * frameStride(kTile, kTile) * kCh[0] + size_t(kSlackPixels) * kCh[kLevels - 1];
  for (Pad *p : {&p_, &p2_}) {
    p->buf = allocate(b_, pad * 2, "VAE encoder padded input");
    std::memset(p->buf.contents(), 0, p->buf.sizeBytes());
  }
  stats_ = allocate(b_, size_t(B) * H3_VAE_ENC_GROUPS * 2 * 4, "VAE encoder GroupNorm stats");
}

void VaeEncoder::loadWeights(const std::string &dir) {
  using splash::model::WeightStoreError;
  const std::string manifestPath = dir + "/manifest_enc.json", blob = dir + "/vae_enc_f16.bin";
  if (!std::filesystem::is_regular_file(manifestPath) || !std::filesystem::is_regular_file(blob))
    throw WeightStoreError(dir + " holds no video VAE encoder (convert_vae.py --encoder)");
  NSDictionary *manifest = model::readManifest(manifestPath);
  NSDictionary *convs = manifest[@"convs"], *norms = manifest[@"norms"], *quant = manifest[@"quant_conv"];
  if (![convs isKindOfClass:[NSDictionary class]] || ![norms isKindOfClass:[NSDictionary class]] ||
      ![quant isKindOfClass:[NSDictionary class]])
    throw WeightStoreError(manifestPath + ": expected convs, norms and quant_conv");
  const model::WeightSource source(blob);
  weights_ = allocate(b_, source.bytes(), "VAE encoder weights");
  model::read(source, 0, source.bytes(), weights_.contents());
  auto field = [&](NSDictionary *record, NSString *key) -> int64_t {
    const std::optional<int64_t> value =
        [record isKindOfClass:[NSDictionary class]] ? model::integer(record[key]) : std::nullopt;
    if (!value) throw WeightStoreError(manifestPath + ": missing " + key.UTF8String);
    return *value;
  };
  auto view = [&](NSDictionary *record, NSString *key) -> BufferView {
    const int64_t offset = field(record, key);
    if (offset < 0 || uint64_t(offset) >= source.bytes())
      throw WeightStoreError(blob + ": bad offset " + key.UTF8String);
    return {weights_, static_cast<uint64_t>(offset)};
  };
  for (NSString *name in convs) {
    NSDictionary *record = convs[name];
    ConvW c;
    c.w = view(record, @"w");
    c.b = view(record, @"b");
    c.cin = static_cast<int32_t>(field(record, @"cin"));
    c.cinPad = static_cast<int32_t>(field(record, @"cin_pad"));
    c.cout = static_cast<int32_t>(field(record, @"cout"));
    c.coutPad = static_cast<int32_t>(field(record, @"cout_pad"));
    c.kd = static_cast<int32_t>(field(record, @"kd"));
    c.k = static_cast<int32_t>(field(record, @"k"));
    convs_[name.UTF8String] = c;
  }
  for (NSString *name in norms) {
    NSDictionary *record = norms[name];
    norms_[name.UTF8String] = NormW{view(record, @"w"), view(record, @"b"), static_cast<int32_t>(field(record, @"c"))};
  }
  quantW_ = view(quant, @"w");
  quantB_ = view(quant, @"b");
  for (int i = 0; i < kLevels; ++i)
    for (int j = 0; j < 2; ++j) {
      const std::string p = "down." + std::to_string(i) + ".block." + std::to_string(j) + ".";
      for (const char *k : {"conv1", "conv2"})
        if (!convs_.count(p + k)) throw WeightStoreError(blob + ": missing " + p + k);
    }
  for (const char *k : {"conv_in", "conv_out"})
    if (!convs_.count(k)) throw WeightStoreError(blob + ": missing " + k);
}

void VaeEncoder::pad(Encoder &e, const Act &src, Pad &dst, int32_t mode, const NormW *norm, int32_t cdst,
                     int32_t frames) {
  dst.H = src.H;
  dst.W = src.W;
  dst.Wp = src.W + 2;
  dst.FS = frameStride(src.H, src.W);
  dst.C = cdst;
  if (norm) {
    const VaeEncStatsParams sp{src.H * src.W, src.C, 0, kEps};
    e.pipeline("vae_enc_stats");
    e.bind(src.buf, 0);
    e.bind(stats_, 1);
    e.bytes(sp, 2);
    e.dispatch(size3(H3_VAE_ENC_GROUPS, frames, 1), size3(256, 1, 1));
  }
  const VaeEncPadParams p{dst.H, dst.W, dst.Wp, dst.FS, src.C, cdst, mode, norm ? 1 : 0};
  e.pipeline("vae_enc_pad");
  e.bind(src.buf, 0);
  e.bind(dst.buf, 1);
  e.bind(stats_, 2);
  e.bind(norm ? norm->w : BufferView{weights_}, 3);
  e.bind(norm ? norm->b : BufferView{weights_}, 4);
  e.bytes(p, 5);
  const int64_t threads = int64_t(dst.FS) * (cdst / 4);
  e.dispatch(size3(static_cast<uint64_t>((threads + 255) / 256), frames, 1), size3(256, 1, 1));
}

void VaeEncoder::conv(Encoder &e, const Pad &src, const ConvW &w, int32_t stride, int32_t tstride, Act &dst,
                      const Act *residual, int32_t frames) {
  const int32_t Hout = stride == 2 ? src.H / 2 : src.H, Wout = stride == 2 ? src.W / 2 : src.W;
  // flat output domain of width Wp: stride 2 reads input row 2 r = 2 y Wp + 2 x (columns x >= Wout are garbage)
  const int32_t Wq = src.Wp, Mv = Hout * Wq;
  const device::Tile t = w.coutPad % 128 ? device::kVaeEncodeTileOut
                                         : (Mv >= device::kVaeEncodeWideRows ? device::kVaeEncodeTileWide
                                                                             : device::kVaeEncodeTileNarrow);
  const int32_t mt = (Mv + t.tm - 1) / t.tm;
  VaeEncConvParams p{};
  p.Wp = src.Wp;
  p.Wq = Wq;
  p.Mv = Mv;
  p.Mrows = mt * t.tm;
  p.Hout = Hout;
  p.Wout = Wout;
  p.Cin = w.cinPad;
  p.Cout = w.cout;
  p.CoutPad = w.coutPad;
  p.FSin = src.FS;
  p.Din = 1;
  p.Dout = 1;
  p.KD = w.kd;
  p.tstride = tstride;
  p.residual = residual ? 1 : 0;
  dst.H = Hout;
  dst.W = Wout;
  dst.C = w.cout;
  const char *kind = w.k == 1 ? "vae_enc_conv1_" : (stride == 2 ? "vae_enc_down_" : "vae_enc_conv3_");
  e.pipeline(kind + tileName(t));
  e.bind(src.buf, 0);
  e.bind(w.w, 1);
  e.bind(w.b, 2);
  e.bind(dst.buf, 3);
  e.bind(residual ? residual->buf : dst.buf, 4);
  e.bytes(p, 5);
  e.dispatch(size3(w.coutPad / t.tn, mt, frames), size3(32 * t.nsg, 1, 1));
}

void VaeEncoder::encodeBatch(Encoder &e, int32_t frames, int32_t tileH, int32_t tileW) {
  pixels_.H = tileH;
  pixels_.W = tileW;
  pixels_.C = 3;
  Act *x = &x_[0], *xo = &x_[1];
  pad(e, pixels_, p_, 0, nullptr, H3_VAE_ENC_IN_CHANNELS, frames);
  conv(e, p_, convs_.at("conv_in"), 1, 1, *x, nullptr, frames);
  for (int i = 0; i < kLevels; ++i) {
    for (int j = 0; j < 2; ++j) {
      const std::string pre = "down." + std::to_string(i) + ".block." + std::to_string(j) + ".";
      pad(e, *x, p_, 0, &norms_.at(pre + "norm1"), x->C, frames);
      conv(e, p_, convs_.at(pre + "conv1"), 1, 1, h_, nullptr, frames);
      pad(e, h_, p_, 0, &norms_.at(pre + "norm2"), h_.C, frames);
      auto nin = convs_.find(pre + "nin_shortcut");
      if (nin != convs_.end()) {
        pad(e, *x, p2_, 0, nullptr, x->C, frames);
        conv(e, p2_, nin->second, 1, 1, s_, nullptr, frames);
        conv(e, p_, convs_.at(pre + "conv2"), 1, 1, s_, &s_, frames);
        std::swap(*x, s_);
      } else {
        conv(e, p_, convs_.at(pre + "conv2"), 1, 1, *x, x, frames);
      }
    }
    if (kSpace[i] * kTime[i] > 1) {
      pad(e, *x, p_, 1, nullptr, x->C, frames);
      conv(e, p_, convs_.at("down." + std::to_string(i) + ".downsample.conv"), kSpace[i], kTime[i], *xo, nullptr,
           frames);
      std::swap(x, xo);
    }
  }
  pad(e, *x, p_, 0, &norms_.at("norm_out"), x->C, frames);
  conv(e, p_, convs_.at("conv_out"), 1, 1, out_, nullptr, frames);
  const VaeEncQuantParams q{frames * out_.H * out_.W, 0, 0, 0};
  e.pipeline("vae_enc_quant");
  e.bind(out_.buf, 0);
  e.bind(quantW_, 1);
  e.bind(quantB_, 2);
  e.bind(moments_.buf, 3);
  e.bytes(q, 4);
  e.dispatch(size3((q.pixels * H3_VAE_ENC_MOMENTS + 255) / 256, 1, 1), size3(256, 1, 1));
  moments_.H = out_.H;
  moments_.W = out_.W;
  moments_.C = H3_VAE_ENC_MOMENTS;
}

void VaeEncoder::encodeRgb(const uint8_t *rgb, int32_t n, int32_t H, int32_t W, std::vector<float> &moments,
                           VaeEncodeStats &stats) {
  static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f}, kStd[3] = {0.229f, 0.224f, 0.225f};
  const size_t count = size_t(n) * H * W * 3;
  std::vector<float> px(count);
  for (size_t i = 0; i < count; ++i) {
    const int c = static_cast<int>(i % 3);
    const float v = static_cast<float>(rgb[i]) / 255.0f;
    const float d = v - kMean[c];
    px[i] = d / kStd[c];
  }
  encodeFrames(px.data(), n, H, W, moments, stats);
}

void VaeEncoder::encodeFrames(const float *pixels, int32_t n, int32_t H, int32_t W, std::vector<float> &moments,
                              VaeEncodeStats &stats) {
  const auto t0 = splash::AwakeClock::now();
  stats = VaeEncodeStats{};
  if (n < 1 || H % kRatio || W % kRatio) throw std::invalid_argument("VAE encode: frames must be multiples of 16");
  std::vector<int32_t> ys, yl, yov, xs, xl, xov;
  splitTiles(H, kTile, kOverlap, ys, yl, yov);
  splitTiles(W, kTile, kOverlap, xs, xl, xov);
  const int32_t th = yl[0], tw = xl[0];
  if (th > capH_ || tw > capW_ || th % 32 || tw % 32)
    throw std::invalid_argument("VAE encode: tile " + std::to_string(th) + "x" + std::to_string(tw) + " unsupported");
  const int32_t ny = static_cast<int32_t>(ys.size()), nx = static_cast<int32_t>(xs.size());
  struct Job {
    int32_t f, i, j;
  };
  std::vector<Job> jobs;
  for (int32_t f = 0; f < n; ++f)
    for (int32_t i = 0; i < ny; ++i)
      for (int32_t j = 0; j < nx; ++j) jobs.push_back({f, i, j});
  const int32_t lh = th / kRatio, lw = tw / kRatio;
  std::vector<Tile> tiles(jobs.size());
  const int32_t B = device::kVaeEncodeBatch;
  for (size_t j0 = 0; j0 < jobs.size(); j0 += B) {
    const int32_t frames = static_cast<int32_t>(std::min<size_t>(B, jobs.size() - j0));
    float *px = static_cast<float *>(pixels_.buf.contents());
    for (int32_t k = 0; k < frames; ++k) {
      const Job &jb = jobs[j0 + k];
      for (int32_t y = 0; y < th; ++y)
        std::memcpy(px + (size_t(k) * th + y) * tw * 3,
                    pixels + ((size_t(jb.f) * H + ys[jb.i] + y) * W + xs[jb.j]) * 3, size_t(tw) * 3 * 4);
    }
    Encoder e(b_);
    encodeBatch(e, frames, th, tw);
    stats.gpuS += run(b_, e);
    ++stats.batches;
    const float *m = static_cast<const float *>(moments_.buf.contents());
    for (int32_t k = 0; k < frames; ++k) {
      Tile &t = tiles[j0 + k];
      t.h = lh;
      t.w = lw;
      t.d.assign(m + size_t(k) * lh * lw * 48, m + size_t(k + 1) * lh * lw * 48);
    }
  }
  stats.tiles = static_cast<int32_t>(jobs.size());
  // VideoVAE._stitch_tiles (overlaps in latent units)
  const int32_t LH = H / kRatio, LW = W / kRatio;
  moments.assign(size_t(n) * LH * LW * 48, 0.0f);
  for (int32_t f = 0; f < n; ++f) {
    auto orig = [&](int32_t i, int32_t j) -> const Tile & { return tiles[(size_t(f) * ny + i) * nx + j]; };
    int32_t y0 = 0;
    for (int32_t i = 0; i < ny; ++i) {
      int32_t x0 = 0, rows = 0;
      for (int32_t j = 0; j < nx; ++j) {
        Tile t = orig(i, j);
        if (i > 0) t = blend(orig(i - 1, j), t, yov[i - 1] / kRatio, 0);
        if (j > 0) t = blend(orig(i, j - 1), t, xov[j - 1] / kRatio, 1);
        const int32_t keepH = i < ny - 1 ? t.h - yov[i] / kRatio : t.h;
        const int32_t keepW = j < nx - 1 ? t.w - xov[j] / kRatio : t.w;
        if (y0 + keepH > LH || x0 + keepW > LW) throw std::invalid_argument("VAE encode: stitch out of range");
        for (int32_t y = 0; y < keepH; ++y)
          std::memcpy(&moments[((size_t(f) * LH + y0 + y) * LW + x0) * 48], &t.d[size_t(y) * t.w * 48],
                      size_t(keepW) * 48 * 4);
        x0 += keepW;
        rows = keepH;
      }
      if (x0 != LW) throw std::invalid_argument("VAE encode: stitched width " + std::to_string(x0));
      y0 += rows;
    }
    if (y0 != LH) throw std::invalid_argument("VAE encode: stitched height " + std::to_string(y0));
  }
  for (float v : moments) stats.nonfinite += !std::isfinite(v);
  stats.wallS = splash::millisecondsSince(t0) / 1e3;
}

} // namespace h3::ops
