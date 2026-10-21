#include "video/h3/model/Prep.hpp"

#include "AwakeClock.hpp"
#include "metal/abi/video/Layout.h"
#include "video/h3/model/Numpy.hpp"
#include "video/h3/model/WeightFiles.hpp"
#include "video/h3/ops/PrepOps.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>

namespace h3::model {
namespace {

using splash::AwakeClock;
double since(AwakeClock::time_point t0) { return splash::millisecondsSince(t0) / 1e3; }

constexpr int32_t kD = H3_DIT_HIDDEN, kTextDim = 5120, kTembIn = 256, kTembHidden = 5376, kTembDim = 2688;
constexpr int32_t kAdaOut = H3_MODALITIES * H3_ADA_ROWS * kD, kFinalOut = 2 * kD;
constexpr int32_t kHeads = H3_DIT_HEADS, kHeadDim = H3_DIT_HEAD_DIM, kInner = H3_DIT_INNER, kFfn = H3_DIT_FFN;
constexpr int32_t kRefinerBlocks = 2;
constexpr float kEps = 1e-5f;  // norm_eps, qk_norm_eps, final_norm_eps

// numpy bf16_bits / from_bf16_bits (h3/common.py): round to nearest even on the fp32 bits
uint16_t bf16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float fromBf16(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// numpy linspace(start, stop, num, endpoint=False): arange(num) * ((stop - start) / num) + start
std::vector<double> linspaceNoEnd(double start, double stop, int num) {
  const double step = (stop - start) / num;
  std::vector<double> y(num);
  for (int i = 0; i < num; ++i) {
    double v = static_cast<double>(i) * step;
    y[i] = v + start;
  }
  return y;
}

// packing._spatial_position_grid
std::vector<double> spatialGrid(int dim, int patch, double sqrtArea) {
  const double ratio = dim / sqrtArea;
  const double left = (1.0 - ratio) / 2.0;
  std::vector<double> g = linspaceNoEnd(left, left + ratio, dim / patch);
  for (double &v : g) v = v * 32;
  return g;
}

// MLX threefry2x32 (random.metal)
void threefry(uint32_t k0, uint32_t k1, uint32_t c0, uint32_t c1, uint32_t &o0, uint32_t &o1) {
  static constexpr uint32_t rot[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
  const uint32_t ks[3] = {k0, k1, k0 ^ k1 ^ 0x1BD11BDAu};
  uint32_t x = c0 + ks[0], y = c1 + ks[1];
  for (int i = 0; i < 5; ++i) {
    for (uint32_t r : rot[i % 2]) {
      x += y;
      y = (y << r) | (y >> (32 - r));
      y ^= x;
    }
    x += ks[(i + 1) % 3];
    y += ks[(i + 2) % 3] + uint32_t(i) + 1u;
  }
  o0 = x;
  o1 = y;
}

// RandomBits with one key: n uint32 values (element y and y + ceil(n / 2) from one hash)
void randomBits(uint32_t k0, uint32_t k1, size_t n, uint32_t *out) {
  const size_t half = n / 2, odd = n % 2, gridY = half + odd;
  dispatch_apply((gridY + 4095) / 4096, DISPATCH_APPLY_AUTO, ^(size_t blk) {
    for (size_t y = blk * 4096; y < std::min(gridY, (blk + 1) * 4096); ++y) {
      const bool dropLast = odd && y == half;
      uint32_t b0, b1;
      threefry(k0, k1, uint32_t(y), dropLast ? 0u : uint32_t(y + gridY), b0, b1);
      out[y] = b0;
      if (!dropLast) out[y + gridY] = b1;
    }
  });
}

// GPU scratch buffers by name
struct Buffers {
  metal::MetalBackend &b;
  std::map<std::string, metal::MetalBuffer> m;
  void make(const std::string &name, std::vector<int64_t> shape, size_t elem) {
    size_t n = 1;
    for (int64_t d : shape) n *= size_t(d);
    m[name] = allocate(b, n * elem, name);
  }
  BufferView operator[](const std::string &name) const { return BufferView{m.at(name)}; }
  template <class T>
  const T *host(const std::string &name) const { return static_cast<const T *>(m.at(name).contents()); }
  void *contents(const std::string &name) const { return m.at(name).contents(); }
};

// A buffer holding the checkpoint's tensor `name`.
metal::MetalBuffer resident(metal::MetalBackend &b, const SafetensorsCheckpoint &checkpoint, const std::string &name,
                            const char *dtype, const std::vector<int64_t> &shape) {
  const SourceTensor &source = tensor(checkpoint, name, dtype, shape);
  metal::MetalBuffer buffer = allocate(b, source.bytes, name);
  read(source, buffer.contents());
  return buffer;
}

std::vector<float> readF32(const SafetensorsCheckpoint &checkpoint, const std::string &name,
                           const std::vector<int64_t> &shape) {
  const SourceTensor &source = tensor(checkpoint, name, "F32", shape);
  std::vector<float> out(source.bytes / 4);
  read(source, out.data());
  return out;
}

// prep.py adaLN LoRA: tab = bf16(fp32(fp64(h) + (st @ A^T) @ B^T)) with numpy's sequential fused accumulation
// (Accelerate dgemm), rows of `n` outputs.
void applyLora(const uint16_t *h, const std::vector<double> &st, const float *A, const float *B, int T, int K, int n,
               std::vector<uint16_t> &tab) {
  tab.resize(size_t(T) * n);
  if (!A) {
    std::memcpy(tab.data(), h, tab.size() * 2);
    return;
  }
  std::vector<double> u(size_t(T) * 16);
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < 16; ++r) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc = std::fma(st[size_t(t) * K + k], double(A[size_t(r) * K + k]), acc);
      u[size_t(t) * 16 + r] = acc;
    }
  const double *up = u.data();
  uint16_t *out = tab.data();
  dispatch_apply((size_t(n) + 4095) / 4096, DISPATCH_APPLY_AUTO, ^(size_t blk) {
    const size_t n0 = blk * 4096, n1 = std::min(size_t(n), n0 + 4096);
    for (int t = 0; t < T; ++t)
      for (size_t j = n0; j < n1; ++j) {
        double dl = 0;
        for (int r = 0; r < 16; ++r) dl = std::fma(up[size_t(t) * 16 + r], double(B[j * 16 + r]), dl);
        const double hf = double(fromBf16(h[size_t(t) * n + j])) + dl;
        out[size_t(t) * n + j] = bf16Bits(static_cast<float>(hf));
      }
  });
}

} // namespace

void Prep::layout(const VideoGeometry &g, int32_t L, Prepared &p) {
  p.frames = g.frames;
  p.textRows = L;
  p.latT = g.latentFrames;
  p.latH = g.latentHeight;
  p.latW = g.latentWidth;
  p.nAud = g.audioRows / 2;
  p.audioRows = g.audioRows;
  p.videoRows = g.videoRows();
  p.rows = L + p.audioRows + p.videoRows;
  const int32_t M = p.rows, n = p.nAud, gh = g.gridH, gw = g.gridW, rowsPerFrame = gh * gw;
  std::vector<double> pos(size_t(M) * 3, 0.0);
  for (int32_t i = 0; i < L; ++i) pos[size_t(i) * 3] = static_cast<double>(i);
  const double sqrtArea = std::sqrt(static_cast<double>(p.latH * p.latW));
  const std::vector<double> hg = spatialGrid(p.latH, 2, sqrtArea), wg = spatialGrid(p.latW, 2, sqrtArea);
  const int32_t audioStart = L, videoStart = L + p.audioRows;
  for (int32_t c = 0; c < 2; ++c)
    for (int32_t t = 0; t < n; ++t) {
      double *r = &pos[size_t(audioStart + c * n + t) * 3];
      r[0] = static_cast<double>(L) + static_cast<double>(t);
      r[2] = c == 0 ? wg.front() : wg.back();
    }
  // _temporal_position_grid: origin + concatenate([0], cumsum(spans[:-1])), spans = 5/3 x (1, 4, 4, 4, 4)[i % 5]
  static constexpr int kPattern[5] = {1, 4, 4, 4, 4};
  const double rescale = 5.0 / 3.0;
  std::vector<double> tgrid(p.latT);
  double acc = 0.0;
  for (int32_t f = 0; f < p.latT; ++f) {
    if (f > 0) acc = acc + rescale * kPattern[(f - 1) % 5];
    tgrid[f] = static_cast<double>(L) + acc;
  }
  for (int32_t f = 0; f < p.latT; ++f)
    for (int32_t y = 0; y < gh; ++y)
      for (int32_t x = 0; x < gw; ++x) {
        double *r = &pos[size_t(videoStart + f * rowsPerFrame + y * gw + x) * 3];
        r[0] = tgrid[f];
        r[1] = hg[y];
        r[2] = wg[x];
      }
  p.pos.resize(pos.size());
  for (size_t i = 0; i < pos.size(); ++i) p.pos[i] = static_cast<float>(pos[i]);
  p.tags.assign(M, 0);
  std::fill(p.tags.begin(), p.tags.begin() + L, uint8_t(1));
  std::fill(p.tags.begin() + audioStart, p.tags.begin() + videoStart, uint8_t(2));
}

void Prep::schedule(int32_t steps, Prepared &p) {
  if (steps < 1) throw std::invalid_argument("prep: steps must be >= 1");
  // scheduler._linspace_1_to_0(steps + 1) and the exponential shift, float32 element ops as numpy
  auto sigmas = [&](float shift) {
    const int n = steps + 1;
    const double step = static_cast<double>(static_cast<float>(-1.0f / static_cast<float>(n - 1)));
    const int half = n / 2;
    std::vector<float> base(n);
    for (int i = 0; i < n; ++i) {
      double v;
      if (i < half) {
        const double si = step * static_cast<double>(i);
        v = 1.0 + si;
      } else {
        const double si = step * static_cast<double>(n - 1 - i);
        v = 0.0 - si;
      }
      base[i] = static_cast<float>(v);
    }
    const float shift32 = shift, shiftM1 = static_cast<float>(static_cast<double>(shift) - 1.0);
    std::vector<float> values;
    for (int i = 0; i < n; ++i) {
      const float num = shift32 * base[i];
      const float den0 = shiftM1 * base[i];
      const float den = 1.0f + den0;
      const float v = num / den;
      if (values.empty() || v != values.back()) values.push_back(v);
    }
    return values;
  };
  p.vSigmas = sigmas(12.0f);
  p.aSigmas = sigmas(3.0f);
  if (int(p.vSigmas.size()) != steps + 1 || int(p.aSigmas.size()) != steps + 1)
    throw std::invalid_argument("prep: the shifted sigma grid collapses below " + std::to_string(steps) + " steps");
  p.steps = steps;
  auto timesteps = [](const std::vector<float> &s) {
    std::vector<float> t(s.size() - 1);
    for (size_t i = 0; i + 1 < s.size(); ++i) t[i] = static_cast<float>(1.0 - static_cast<double>(s[i]));
    return t;
  };
  p.vT = timesteps(p.vSigmas);
  p.aT = timesteps(p.aSigmas);
  std::set<float> distinct(p.vT.begin(), p.vT.end());
  distinct.insert(p.aT.begin(), p.aT.end());
  p.table.assign(distinct.begin(), distinct.end());
  auto index = [&](const std::vector<float> &t) {
    std::vector<int64_t> idx(t.size());
    for (size_t i = 0; i < t.size(); ++i)
      idx[i] = std::lower_bound(p.table.begin(), p.table.end(), t[i]) - p.table.begin();
    return idx;
  };
  p.tv = index(p.vT);
  p.ta = index(p.aT);
}

void Prep::noiseBits(uint64_t seed, size_t videoCount, size_t audioCount, std::vector<uint32_t> &video,
                     std::vector<uint32_t> &audio) {
  // mx.random.seed(seed): key (seed >> 32, seed); every draw takes KeySequence::next(): split(key) -> key = [0],
  // subkey = [1]
  uint32_t k0 = static_cast<uint32_t>(seed >> 32), k1 = static_cast<uint32_t>(seed);
  auto next = [&](uint32_t &s0, uint32_t &s1) {
    uint32_t out[4];
    randomBits(k0, k1, 4, out);
    k0 = out[0];
    k1 = out[1];
    s0 = out[2];
    s1 = out[3];
  };
  uint32_t s0, s1;
  next(s0, s1);
  video.resize(videoCount);
  randomBits(s0, s1, videoCount, video.data());
  next(s0, s1);
  audio.resize(audioCount);
  randomBits(s0, s1, audioCount, audio.data());
}

void Prep::run(metal::MetalBackend &b, const std::string &weights, Variant variant, const VideoGeometry &g,
               const PrepInputs &in, Prepared &p) {
  const auto t0 = AwakeClock::now();
  if (!in.hidden || in.textRows < 2) throw std::invalid_argument("prep: text hidden states need >= 2 rows");
  const int32_t L = in.textRows;
  const int32_t steps = in.steps > 0 ? in.steps : (variant == Variant::Turbo ? 8 : 50);
  layout(g, L, p);
  schedule(steps, p);
  p.layoutS = since(t0);
  const std::string ditDir = weights + (variant == Variant::Turbo ? "/dit-turbo" : "/dit-default");
  const SafetensorsCheckpoint dit(ditDir);
  ops::PrepOps op;
  const int32_t T = static_cast<int32_t>(p.table.size()), M = p.rows;
  Buffers buf{b, {}};
  // ---- one command: RoPE, timestep MLP, final adaLN base, text path, noise
  const auto t1 = AwakeClock::now();
  buf.make("inv_freq", {16}, 4);
  buf.make("pos", {M, 3}, 4);
  buf.make("cos", {M, 48}, 4);
  buf.make("sin", {M, 48}, 4);
  std::memcpy(buf.contents("pos"), p.pos.data(), p.pos.size() * 4);
  buf.make("table", {T}, 4);
  std::memcpy(buf.contents("table"), p.table.data(), size_t(T) * 4);
  buf.make("te_sin", {T, kTembIn}, 4);
  buf.make("te_h", {T, kTembHidden}, 4);
  buf.make("te_s", {T, kTembHidden}, 4);
  buf.make("temb", {T, kTembDim}, 4);
  buf.make("st", {T, kTembDim}, 4);
  buf.make("st_bf", {T, kTembDim}, 2);
  buf.make("fin", {T, kFinalOut}, 2);
  buf.make("hidden", {L, kTextDim}, 4);
  std::memcpy(buf.contents("hidden"), in.hidden, size_t(L) * kTextDim * 4);
  buf.make("x0", {L, kTextDim}, 2);
  buf.make("cp", {L, kD}, 2);
  for (int blk = 0; blk < kRefinerBlocks; ++blk) {
    const std::string s = "b" + std::to_string(blk) + "_";
    buf.make(s + "n1", {L, kD}, 2);
    buf.make(s + "qkv", {L, 3 * kInner}, 2);
    buf.make(s + "qn", {L * kHeads, kHeadDim}, 2);
    buf.make(s + "kn", {L * kHeads, kHeadDim}, 2);
    buf.make(s + "att", {L, kInner}, 2);
    buf.make(s + "o", {L, kD}, 2);
    buf.make(s + "x1", {L, kD}, 2);
    buf.make(s + "n2", {L, kD}, 2);
    buf.make(s + "f1", {L, 2 * kFfn}, 2);
    buf.make(s + "act", {L, kFfn}, 2);
    buf.make(s + "m", {L, kD}, 2);
    buf.make(s + "x2", {L, kD}, 2);
  }
  buf.make("final", {L, kD}, 2);
  const size_t videoCount = size_t(H3_VAE_LATENT) * p.latT * p.latH * p.latW, audioCount = size_t(p.audioRows) * 32;
  std::vector<uint32_t> vbits, abits;
  noiseBits(in.seed, videoCount, audioCount, vbits, abits);
  buf.make("bits_v", {int64_t(videoCount)}, 4);
  buf.make("bits_a", {int64_t(audioCount)}, 4);
  std::memcpy(buf.contents("bits_v"), vbits.data(), videoCount * 4);
  std::memcpy(buf.contents("bits_a"), abits.data(), audioCount * 4);
  buf.make("noise_v", {1, H3_VAE_LATENT, p.latT, p.latH, p.latW}, 4);
  buf.make("noise_a", {p.audioRows, 32}, 4);

  auto weight = [&](const std::string &name, const char *dtype, const std::vector<int64_t> &shape) {
    return BufferView{resident(b, dit, name, dtype, shape)};
  };
  const BufferView teInW = weight("time_embedder.proj_in.weight", "F32", {kTembHidden, kTembIn});
  const BufferView teInB = weight("time_embedder.proj_in.bias", "F32", {kTembHidden});
  const BufferView teOutW = weight("time_embedder.proj_out.weight", "F32", {kTembDim, kTembHidden});
  const BufferView teOutB = weight("time_embedder.proj_out.bias", "F32", {kTembDim});
  const BufferView finW = weight("final_layer.adaln_proj.linear.weight", "BF16", {kFinalOut, kTembDim});
  const BufferView finB = weight("final_layer.adaln_proj.linear.bias", "BF16", {kFinalOut});
  const BufferView cpW = weight("condition_proj.weight", "BF16", {kD, kTextDim});
  const BufferView cpB = weight("condition_proj.bias", "BF16", {kD});
  const BufferView finalNorm = weight("token_refiner.final_norm.weight", "BF16", {kD});
  struct RefinerBlock {
    BufferView norm1, qkv, qNorm, kNorm, out, norm2, fc1, fc2;
  } rb[kRefinerBlocks];
  for (int blk = 0; blk < kRefinerBlocks; ++blk) {
    const std::string s = "token_refiner.blocks." + std::to_string(blk) + ".";
    rb[blk].norm1 = weight(s + "norm1.weight", "BF16", {kD});
    rb[blk].qkv = weight(s + "attn.qkv_proj.weight", "BF16", {3 * kInner, kD});
    rb[blk].qNorm = weight(s + "attn.q_norm.weight", "BF16", {kHeadDim});
    rb[blk].kNorm = weight(s + "attn.k_norm.weight", "BF16", {kHeadDim});
    rb[blk].out = weight(s + "attn.out_proj.weight", "BF16", {kD, kInner});
    rb[blk].norm2 = weight(s + "norm2.weight", "BF16", {kD});
    rb[blk].fc1 = weight(s + "mlp.fc1.weight", "BF16", {2 * kFfn, kD});
    rb[blk].fc2 = weight(s + "mlp.fc2.weight", "BF16", {kD, kFfn});
  }
  const float attnScale = static_cast<float>(std::pow(double(kHeadDim), -0.5));
  {
    Encoder e(b);
    op.ropeInvFreq(e, buf["inv_freq"]);
    op.ropeTrig(e, buf["pos"], buf["inv_freq"], buf["cos"], buf["sin"], M);
    op.tembSinusoid(e, buf["table"], buf["te_sin"], T);
    op.linear(e, true, buf["te_sin"], teInW, teInB, buf["te_h"], T, kTembHidden, kTembIn);
    op.siluF32(e, buf["te_h"], buf["te_s"], T * kTembHidden);
    op.linear(e, true, buf["te_s"], teOutW, teOutB, buf["temb"], T, kTembDim, kTembHidden);
    op.siluF32(e, buf["temb"], buf["st"], T * kTembDim);
    op.castBf16(e, buf["st"], buf["st_bf"], T * kTembDim);
    op.linear(e, false, buf["st_bf"], finW, finB, buf["fin"], T, kFinalOut, kTembDim);
    // text path
    op.castBf16(e, buf["hidden"], buf["x0"], L * kTextDim);
    op.linear(e, false, buf["x0"], cpW, cpB, buf["cp"], L, kD, kTextDim);
    std::string x = "cp";
    for (int blk = 0; blk < kRefinerBlocks; ++blk) {
      const std::string s = "b" + std::to_string(blk) + "_";
      const RefinerBlock &r = rb[blk];
      op.rms(e, buf[x], r.norm1, buf[s + "n1"], L, kD, kD, kD, kEps);
      op.linear(e, false, buf[s + "n1"], r.qkv, {}, buf[s + "qkv"], L, 3 * kInner, kD);
      op.rms(e, buf[s + "qkv"], r.qNorm, buf[s + "qn"], L * kHeads, kHeadDim, 3 * kHeadDim, kHeadDim, kEps);
      op.rms(e, buf[s + "qkv"] + size_t(kHeadDim) * 2, r.kNorm, buf[s + "kn"], L * kHeads, kHeadDim, 3 * kHeadDim,
             kHeadDim, kEps);
      const ops::PrepOps::AttnStrides st{kHeadDim, kInner, kHeadDim, kInner, 3 * kHeadDim, 3 * kInner, kHeadDim, kInner};
      op.attention(e, buf[s + "qn"], buf[s + "kn"], buf[s + "qkv"] + size_t(2 * kHeadDim) * 2, buf[s + "att"], L,
                   kHeads, attnScale, st);
      op.linear(e, false, buf[s + "att"], r.out, {}, buf[s + "o"], L, kD, kInner);
      op.add(e, buf[x], buf[s + "o"], buf[s + "x1"], L * kD);
      op.rms(e, buf[s + "x1"], r.norm2, buf[s + "n2"], L, kD, kD, kD, kEps);
      op.linear(e, false, buf[s + "n2"], r.fc1, {}, buf[s + "f1"], L, 2 * kFfn, kD);
      op.swiglu(e, buf[s + "f1"], buf[s + "act"], L, kFfn);
      op.linear(e, false, buf[s + "act"], r.fc2, {}, buf[s + "m"], L, kD, kFfn);
      op.add(e, buf[s + "x1"], buf[s + "m"], buf[s + "x2"], L * kD);
      x = s + "x2";
    }
    op.rms(e, buf[x], finalNorm, buf["final"], L, kD, kD, kD, kEps);
    op.normal(e, buf["bits_v"], buf["noise_v"], int(videoCount));
    op.normal(e, buf["bits_a"], buf["noise_a"], int(audioCount));
    h3::run(b, e);
  }
  p.textS = since(t1);
  // RoPE bf16 [M][96] = cos 48 | sin 48
  {
    const float *cs = buf.host<float>("cos"), *sn = buf.host<float>("sin");
    p.rope.resize(size_t(M) * 96);
    for (size_t r = 0; r < size_t(M); ++r)
      for (int c = 0; c < 48; ++c) {
        p.rope[r * 96 + c] = bf16Bits(cs[r * 48 + c]);
        p.rope[r * 96 + 48 + c] = bf16Bits(sn[r * 48 + c]);
      }
  }
  p.text.assign(buf.host<uint16_t>("final"), buf.host<uint16_t>("final") + size_t(L) * kD);
  {  // noise: video latents [24][F][H][W] -> patchified rows [(f, gy, gx)][c * 4 + py * 2 + px], then audio rows
    const float *v = buf.host<float>("noise_v");
    const int32_t F = p.latT, Hh = p.latH, Ww = p.latW, gh = Hh / 2, gw = Ww / 2;
    p.videoNoise.resize(size_t(p.videoRows) * 96);
    for (int32_t f = 0; f < F; ++f)
      for (int32_t y = 0; y < gh; ++y)
        for (int32_t x = 0; x < gw; ++x)
          for (int32_t c = 0; c < 24; ++c)
            for (int32_t py = 0; py < 2; ++py)
              for (int32_t px = 0; px < 2; ++px)
                p.videoNoise[((size_t(f) * gh + y) * gw + x) * 96 + c * 4 + py * 2 + px] =
                    v[((size_t(c) * F + f) * Hh + 2 * y + py) * Ww + 2 * x + px];
    p.audioNoise.assign(buf.host<float>("noise_a"), buf.host<float>("noise_a") + audioCount);
  }
  // ---- adaLN tables
  const auto t2 = AwakeClock::now();
  std::vector<double> st(size_t(T) * kTembDim);
  for (size_t i = 0; i < st.size(); ++i) st[i] = double(buf.host<float>("st")[i]);
  p.adaBlocks.assign(size_t(steps) * H3_DIT_BLOCKS * kAdaOut, 0);
  p.adaFinal.assign(size_t(steps) * H3_MODALITIES * kFinalOut, 0);
  {  // final layer (vdn-globals LoRA)
    std::vector<float> A, B;
    p.finalLora = dit.find("norm_out.linear.lora_A") != nullptr;
    if (p.finalLora) {
      A = readF32(dit, "norm_out.linear.lora_A", {16, kTembDim});
      B = readF32(dit, "norm_out.linear.lora_B", {kFinalOut, 16});
    }
    std::vector<uint16_t> tab;
    applyLora(buf.host<uint16_t>("fin"), st, p.finalLora ? A.data() : nullptr, p.finalLora ? B.data() : nullptr, T,
              kTembDim, kFinalOut, tab);
    for (int32_t s = 0; s < steps; ++s)
      for (int m = 0; m < H3_MODALITIES; ++m) {
        const int64_t t = m < 2 ? p.tv[s] : p.ta[s];
        std::memcpy(&p.adaFinal[(size_t(s) * H3_MODALITIES + m) * kFinalOut], &tab[size_t(t) * kFinalOut], kFinalOut * 2);
      }
  }
  // block bases from the FL2VA shards, two slots: block i + 1 is read while the GPU evaluates block i
  const SafetensorsCheckpoint transformer(weights + "/fl2va/transformer");
  metal::MetalBuffer wslot[2], bslot[2], hslot[2];
  for (int s = 0; s < 2; ++s) {
    wslot[s] = allocate(b, size_t(kAdaOut) * kTembDim * 2, "adaLN base slot");
    bslot[s] = allocate(b, size_t(kAdaOut) * 2, "adaLN bias slot");
    hslot[s] = allocate(b, size_t(T) * kAdaOut * 2, "adaLN output slot");
  }
  auto readBlock = [&](int i, int s) {
    const std::string k = "blocks." + std::to_string(i) + ".adaln_proj.linear.";
    const SourceTensor &wt = tensor(transformer, k + "weight", "BF16", {kAdaOut, kTembDim});
    const SourceTensor &bt = tensor(transformer, k + "bias", "BF16", {kAdaOut});
    read(bt, bslot[s].contents());
    read(wt, wslot[s].contents());
    p.adalnReadGb += (wt.bytes + bt.bytes) / 1e9;
  };
  readBlock(0, 0);
  std::vector<uint16_t> tab;
  for (int i = 0; i < H3_DIT_BLOCKS; ++i) {
    const int s = i & 1;
    Encoder e(b);
    op.linear(e, false, buf["st_bf"], wslot[s], bslot[s], hslot[s], T, kAdaOut, kTembDim);
    metal::CommandTicket ticket = submit(b, e);
    if (i + 1 < H3_DIT_BLOCKS) readBlock(i + 1, s ^ 1);
    const std::string ka = "blocks." + std::to_string(i) + ".adaln_proj.lora_A";
    std::vector<float> A, B;
    const bool lora = dit.find(ka) != nullptr;
    if (lora) {
      A = readF32(dit, ka, {16, kTembDim});
      B = readF32(dit, "blocks." + std::to_string(i) + ".adaln_proj.lora_B", {kAdaOut, 16});
      ++p.adalnLoraBlocks;
    }
    wait(ticket);
    applyLora(static_cast<const uint16_t *>(hslot[s].contents()), st, lora ? A.data() : nullptr,
              lora ? B.data() : nullptr, T, kTembDim, kAdaOut, tab);
    constexpr size_t kTag = size_t(H3_ADA_ROWS) * kD;
    for (int32_t sp = 0; sp < steps; ++sp)
      for (int m = 0; m < H3_MODALITIES; ++m) {
        const int64_t t = m < 2 ? p.tv[sp] : p.ta[sp];
        std::memcpy(&p.adaBlocks[((size_t(sp) * H3_DIT_BLOCKS + i) * H3_MODALITIES + m) * kTag],
                    &tab[size_t(t) * kAdaOut + m * kTag], kTag * 2);
      }
  }
  p.adalnS = since(t2);
  p.totalS = since(t0);
}

void Prep::rope(metal::MetalBackend &b, Prepared &p) {
  ops::PrepOps op;
  const int32_t M = p.rows;
  Buffers buf{b, {}};
  buf.make("inv_freq", {16}, 4);
  buf.make("pos", {M, 3}, 4);
  buf.make("cos", {M, 48}, 4);
  buf.make("sin", {M, 48}, 4);
  std::memcpy(buf.contents("pos"), p.pos.data(), p.pos.size() * 4);
  Encoder e(b);
  op.ropeInvFreq(e, buf["inv_freq"]);
  op.ropeTrig(e, buf["pos"], buf["inv_freq"], buf["cos"], buf["sin"], M);
  h3::run(b, e);
  const float *cs = buf.host<float>("cos"), *sn = buf.host<float>("sin");
  p.rope.resize(size_t(M) * 96);
  for (size_t r = 0; r < size_t(M); ++r)
    for (int c = 0; c < 48; ++c) {
      p.rope[r * 96 + c] = bf16Bits(cs[r * 48 + c]);
      p.rope[r * 96 + 48 + c] = bf16Bits(sn[r * 48 + c]);
    }
}

void Prep::write(const std::string &dir, const Prepared &p, Variant variant, uint64_t seed, const std::string &source) {
  std::filesystem::create_directories(dir);
  auto i64 = [](int64_t v) { return NpyArray::of("<i8", {}, &v, 8); };
  const int32_t M = p.rows;
  const NpyArray tags = NpyArray::of("|u1", {M}, p.tags.data(), p.tags.size());
  const NpyArray rope = NpyArray::of("<u2", {M, 96}, p.rope.data(), p.rope.size() * 2);
  const NpyArray pos = NpyArray::of("<f4", {M, 3}, p.pos.data(), p.pos.size() * 4);
  const NpyArray L = i64(p.textRows), Na = i64(p.audioRows), Nv = i64(p.videoRows), latT = i64(p.latT),
                 latH = i64(p.latH), latW = i64(p.latW), nAud = i64(p.nAud), frames = i64(p.frames);
  writeNpz(dir + "/layout.npz", {{"tags", &tags}, {"rope", &rope}, {"pos", &pos}, {"L", &L}, {"Na", &Na}, {"Nv", &Nv},
                                 {"lat_t", &latT}, {"lat_h", &latH}, {"lat_w", &latW}, {"n_aud", &nAud},
                                 {"frames", &frames}});
  const int64_t S = p.steps, T = static_cast<int64_t>(p.table.size());
  const NpyArray vs = NpyArray::of("<f4", {S + 1}, p.vSigmas.data(), (S + 1) * 4);
  const NpyArray as = NpyArray::of("<f4", {S + 1}, p.aSigmas.data(), (S + 1) * 4);
  const NpyArray vt = NpyArray::of("<f4", {S}, p.vT.data(), S * 4), at = NpyArray::of("<f4", {S}, p.aT.data(), S * 4);
  const NpyArray table = NpyArray::of("<f4", {T}, p.table.data(), T * 4);
  const NpyArray tv = NpyArray::of("<i8", {S}, p.tv.data(), S * 8), ta = NpyArray::of("<i8", {S}, p.ta.data(), S * 8);
  writeNpz(dir + "/sched.npz", {{"v_sigmas", &vs}, {"a_sigmas", &as}, {"v_t", &vt}, {"a_t", &at}, {"table", &table},
                                {"tv", &tv}, {"ta", &ta}});
  writeNpy(dir + "/ada_blocks.npy", NpyArray::of("<u2", {S, H3_DIT_BLOCKS, H3_MODALITIES, H3_ADA_ROWS, kD},
                                                 p.adaBlocks.data(), p.adaBlocks.size() * 2));
  writeNpy(dir + "/ada_final.npy",
           NpyArray::of("<u2", {S, H3_MODALITIES, 2, kD}, p.adaFinal.data(), p.adaFinal.size() * 2));
  writeNpy(dir + "/text_ref.npy", NpyArray::of("<u2", {p.textRows, kD}, p.text.data(), p.text.size() * 2));
  const NpyArray video = NpyArray::of("<f4", {p.videoRows, 96}, p.videoNoise.data(), p.videoNoise.size() * 4);
  const NpyArray audio = NpyArray::of("<f4", {p.audioRows, 32}, p.audioNoise.data(), p.audioNoise.size() * 4);
  writeNpz(dir + "/noise.npz", {{"video", &video}, {"audio", &audio}});
  auto list = [](const std::vector<float> &v) {
    std::string s = "[";
    char x[32];
    for (size_t i = 0; i < v.size(); ++i) {
      snprintf(x, sizeof x, "%s%.9g", i ? ", " : "", double(v[i]));
      s += x;
    }
    return s + "]";
  };
  char json[4096];
  snprintf(json, sizeof json,
           "{\n \"M\": %d,\n \"L\": %d,\n \"Na\": %d,\n \"Nv\": %d,\n \"lat_t\": %d,\n \"frames\": %d,\n \"steps\": %d,\n"
           " \"distinct_t\": %lld,\n \"seed\": %llu,\n \"variant\": \"%s\",\n \"adaln_lora_blocks\": %d,\n"
           " \"final_lora\": %s,\n \"prompt\": \"%s\",\n \"v_sigmas\": %s,\n \"a_sigmas\": %s,\n \"native\": true,\n"
           " \"layout_s\": %.3f,\n \"text_s\": %.3f,\n \"adaln_s\": %.3f,\n \"adaln_read_GB\": %.2f,\n \"total_s\": %.3f\n}",
           M, p.textRows, p.audioRows, p.videoRows, p.latT, p.frames, p.steps, static_cast<long long>(T),
           static_cast<unsigned long long>(seed), variant == Variant::Turbo ? "turbo" : "default", p.adalnLoraBlocks,
           p.finalLora ? "true" : "false", source.c_str(), list(p.vSigmas).c_str(), list(p.aSigmas).c_str(), p.layoutS,
           p.textS, p.adalnS, p.adalnReadGb, p.totalS);
  FILE *f = fopen((dir + "/prep.json").c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + dir + "/prep.json");
  fputs(json, f);
  fclose(f);
}

} // namespace h3::model
