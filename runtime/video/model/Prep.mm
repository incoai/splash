#include "video/model/Prep.hpp"

#include "AwakeClock.hpp"
#include "metal/abi/video/Layout.h"
#include "video/model/Latents.hpp"
#include "video/model/VaeWeights.hpp"
#include "video/model/WeightFiles.hpp"
#include "video/ops/PrepOps.hpp"
#include "video/ops/VaeEncoder.hpp"

#include <dispatch/dispatch.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <set>
#include <stdexcept>

namespace splash::video::model {
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

// numpy add.reduce of float64 (pairwise summation, 8 accumulators, blocks of 128) from the identity 0
double pairwiseSum(const double *a, int64_t n) {
  if (n < 8) {
    double r = 0.0;
    for (int64_t i = 0; i < n; ++i) r += a[i];
    return r;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int64_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
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

int32_t variantSteps(Variant variant) { return variant == Variant::Turbo ? 8 : 50; }

// The tensors of adaln.safetensors, by schedule: without keyframes (text) or with them (keyframes).
std::string adalnName(bool keyframes, const char *what) {
  return std::string("adaln.") + (keyframes ? "keyframes." : "text.") + what;
}

// A safetensors file written in place: its header first, then each tensor's bytes at its offset, published by rename.
class SafetensorsWriter {
public:
  struct Entry {
    std::string name, dtype;
    std::vector<int64_t> shape;
    uint64_t bytes = 0, offset = 0;
  };

  SafetensorsWriter(std::string path, std::vector<Entry> entries) : path_(std::move(path)), entries_(std::move(entries)) {
    std::string header = "{";
    uint64_t offset = 0;
    for (Entry &e : entries_) {
      e.offset = offset;
      offset += e.bytes;
      std::string shape;
      for (int64_t d : e.shape) shape += (shape.empty() ? "" : ",") + std::to_string(d);
      header += std::format("{}\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                            header.size() > 1 ? "," : "", e.name, e.dtype, shape, e.offset, offset);
    }
    header += "}";
    header.append((8 - header.size() % 8) % 8, ' ');
    data_ = 8 + header.size();
    file_ = std::fopen((path_ + ".partial").c_str(), "wb");
    if (!file_) throw std::runtime_error("cannot write " + path_ + ".partial");
    const uint64_t length = header.size();
    write(0, &length, 8);
    write(8, header.data(), header.size());
  }
  ~SafetensorsWriter() {
    if (file_) std::fclose(file_);
  }
  SafetensorsWriter(const SafetensorsWriter &) = delete;
  SafetensorsWriter &operator=(const SafetensorsWriter &) = delete;

  // Bytes at `at` of tensor `index`.
  void put(size_t index, uint64_t at, const void *bytes, uint64_t size) {
    const Entry &e = entries_.at(index);
    if (at + size > e.bytes) throw std::logic_error(e.name + ": write past its end");
    write(data_ + e.offset + at, bytes, size);
  }
  void publish() {
    const bool closed = std::fflush(file_) == 0 && fsync(fileno(file_)) == 0;
    std::fclose(file_);
    file_ = nullptr;
    if (!closed || std::rename((path_ + ".partial").c_str(), path_.c_str()) != 0)
      throw std::runtime_error("cannot write " + path_);
  }

private:
  void write(uint64_t at, const void *bytes, uint64_t size) {
    if (fseeko(file_, static_cast<off_t>(at), SEEK_SET) != 0 || std::fwrite(bytes, 1, size, file_) != size)
      throw std::runtime_error("cannot write " + path_ + ".partial");
  }

  std::string path_;
  std::vector<Entry> entries_;
  uint64_t data_ = 0;
  std::FILE *file_ = nullptr;
};

} // namespace

void Prep::tabulateAdaLn(metal::MetalBackend &b, const std::string &ditDir, const std::string &transformerDir,
                         Variant variant) {
  const SafetensorsCheckpoint dit(ditDir), transformer(transformerDir);
  const int32_t steps = variantSteps(variant);
  // the schedule without keyframes and with them, whose distinct timesteps index the tables
  struct Schedule {
    Prepared p;
    int32_t T = 0;
    Buffers buf;
    std::vector<double> st;
    metal::MetalBuffer out[2];
  };
  Schedule sched[2] = {{{}, 0, {b, {}}, {}, {}}, {{}, 0, {b, {}}, {}, {}}};
  std::vector<SafetensorsWriter::Entry> entries;
  for (int k = 0; k < 2; ++k) {
    Schedule &s = sched[k];
    schedule(steps, k == 1, s.p);
    s.T = static_cast<int32_t>(s.p.table.size());
    const uint64_t T = uint64_t(s.T);
    entries.push_back({adalnName(k == 1, "table"), "F32", {s.T}, T * 4});
    entries.push_back({adalnName(k == 1, "final"), "BF16", {s.T, kFinalOut}, T * kFinalOut * 2});
    entries.push_back({adalnName(k == 1, "blocks"), "BF16", {H3_DIT_BLOCKS, s.T, kAdaOut}, H3_DIT_BLOCKS * T * kAdaOut * 2});
  }
  SafetensorsWriter file(ditDir + "/adaln.safetensors", entries);
  auto weight = [&](const std::string &name, const char *dtype, const std::vector<int64_t> &shape) {
    return BufferView{resident(b, dit, name, dtype, shape)};
  };
  const BufferView teInW = weight("time_embedder.proj_in.weight", "F32", {kTembHidden, kTembIn});
  const BufferView teInB = weight("time_embedder.proj_in.bias", "F32", {kTembHidden});
  const BufferView teOutW = weight("time_embedder.proj_out.weight", "F32", {kTembDim, kTembHidden});
  const BufferView teOutB = weight("time_embedder.proj_out.bias", "F32", {kTembDim});
  const BufferView finW = weight("final_layer.adaln_proj.linear.weight", "BF16", {kFinalOut, kTembDim});
  const BufferView finB = weight("final_layer.adaln_proj.linear.bias", "BF16", {kFinalOut});
  ops::PrepOps op;
  {  // timestep MLP and the final layer's adaLN base, of each schedule's distinct timesteps
    Encoder e(b);
    for (Schedule &s : sched) {
      const int32_t T = s.T;
      Buffers &buf = s.buf;
      buf.make("table", {T}, 4);
      std::memcpy(buf.contents("table"), s.p.table.data(), size_t(T) * 4);
      buf.make("te_sin", {T, kTembIn}, 4);
      buf.make("te_h", {T, kTembHidden}, 4);
      buf.make("te_s", {T, kTembHidden}, 4);
      buf.make("temb", {T, kTembDim}, 4);
      buf.make("st", {T, kTembDim}, 4);
      buf.make("st_bf", {T, kTembDim}, 2);
      buf.make("fin", {T, kFinalOut}, 2);
      for (metal::MetalBuffer &out : s.out) out = allocate(b, size_t(T) * kAdaOut * 2, "adaLN output slot");
      op.tembSinusoid(e, buf["table"], buf["te_sin"], T);
      op.linear(e, true, buf["te_sin"], teInW, teInB, buf["te_h"], T, kTembHidden, kTembIn);
      op.siluF32(e, buf["te_h"], buf["te_s"], T * kTembHidden);
      op.linear(e, true, buf["te_s"], teOutW, teOutB, buf["temb"], T, kTembDim, kTembHidden);
      op.siluF32(e, buf["temb"], buf["st"], T * kTembDim);
      op.castBf16(e, buf["st"], buf["st_bf"], T * kTembDim);
      op.linear(e, false, buf["st_bf"], finW, finB, buf["fin"], T, kFinalOut, kTembDim);
    }
    video::run(b, e);
  }
  // final layer (vdn-globals LoRA)
  const bool finalLora = dit.find("norm_out.linear.lora_A") != nullptr;
  std::vector<float> finalA, finalB;
  if (finalLora) {
    finalA = readF32(dit, "norm_out.linear.lora_A", {16, kTembDim});
    finalB = readF32(dit, "norm_out.linear.lora_B", {kFinalOut, 16});
  }
  std::vector<uint16_t> tab;
  for (int k = 0; k < 2; ++k) {
    Schedule &s = sched[k];
    s.st.resize(size_t(s.T) * kTembDim);
    for (size_t i = 0; i < s.st.size(); ++i) s.st[i] = double(s.buf.host<float>("st")[i]);
    applyLora(s.buf.host<uint16_t>("fin"), s.st, finalLora ? finalA.data() : nullptr,
              finalLora ? finalB.data() : nullptr, s.T, kTembDim, kFinalOut, tab);
    file.put(3 * k, 0, s.p.table.data(), uint64_t(s.T) * 4);
    file.put(3 * k + 1, 0, tab.data(), tab.size() * 2);
  }
  // block bases from the FL2VA shards, two slots: block i + 1 is read while the GPU evaluates block i
  metal::MetalBuffer wslot[2], bslot[2];
  for (int s = 0; s < 2; ++s) {
    wslot[s] = allocate(b, size_t(kAdaOut) * kTembDim * 2, "adaLN base slot");
    bslot[s] = allocate(b, size_t(kAdaOut) * 2, "adaLN bias slot");
  }
  auto readBlock = [&](int i, int s) {
    const std::string k = "blocks." + std::to_string(i) + ".adaln_proj.linear.";
    read(tensor(transformer, k + "bias", "BF16", {kAdaOut}), bslot[s].contents());
    read(tensor(transformer, k + "weight", "BF16", {kAdaOut, kTembDim}), wslot[s].contents());
  };
  readBlock(0, 0);
  for (int i = 0; i < H3_DIT_BLOCKS; ++i) {
    const int slot = i & 1;
    Encoder e(b);
    for (Schedule &s : sched)
      op.linear(e, false, s.buf["st_bf"], wslot[slot], bslot[slot], s.out[slot], s.T, kAdaOut, kTembDim);
    metal::CommandTicket ticket = submit(b, e);
    if (i + 1 < H3_DIT_BLOCKS) readBlock(i + 1, slot ^ 1);
    const std::string ka = "blocks." + std::to_string(i) + ".adaln_proj.lora_A";
    std::vector<float> A, B;
    const bool lora = dit.find(ka) != nullptr;
    if (lora) {
      A = readF32(dit, ka, {16, kTembDim});
      B = readF32(dit, "blocks." + std::to_string(i) + ".adaln_proj.lora_B", {kAdaOut, 16});
    }
    wait(ticket);
    for (int k = 0; k < 2; ++k) {
      Schedule &s = sched[k];
      applyLora(static_cast<const uint16_t *>(s.out[slot].contents()), s.st, lora ? A.data() : nullptr,
                lora ? B.data() : nullptr, s.T, kTembDim, kAdaOut, tab);
      file.put(3 * k + 2, uint64_t(i) * tab.size() * 2, tab.data(), tab.size() * 2);
    }
  }
  file.publish();
}

void Prep::layout(const ops::VideoGeometry &g, int32_t L, const uint8_t *textTags, const std::vector<int32_t> &anchors,
                  Prepared &p) {
  p.frames = g.frames;
  p.textRows = L;
  p.latT = g.latentFrames;
  p.latH = g.latentHeight;
  p.latW = g.latentWidth;
  p.nAud = g.audioRows / 2;
  p.audioRows = g.audioRows;
  p.videoRows = g.videoRows();
  p.anchors = anchors;
  p.condRows = static_cast<int32_t>(anchors.size()) * g.tokensPerFrame;
  p.rows = L + p.condRows + p.audioRows + p.videoRows;
  const int32_t M = p.rows, n = p.nAud, gh = g.gridH, gw = g.gridW, rowsPerFrame = gh * gw;
  std::vector<double> pos(size_t(M) * 3, 0.0);
  for (int32_t i = 0; i < L; ++i) pos[size_t(i) * 3] = static_cast<double>(i);
  const double sqrtArea = std::sqrt(static_cast<double>(p.latH * p.latW));
  const std::vector<double> hg = spatialGrid(p.latH, 2, sqrtArea), wg = spatialGrid(p.latW, 2, sqrtArea);
  const int32_t condStart = L, audioStart = L + p.condRows, videoStart = audioStart + p.audioRows;
  for (size_t k = 0; k < anchors.size(); ++k) {
    // anchor time: the first latent frame (L), or the last one: L + _temporal_position_span(F) - 5/3, the span summed
    // pairwise by numpy
    double t = static_cast<double>(L);
    if (anchors[k] == 1) {
      static constexpr int kPat[5] = {1, 4, 4, 4, 4};
      std::vector<double> spans(p.latT, 5.0 / 3.0);
      for (int32_t f = 0; f < p.latT; ++f) spans[f] = spans[f] * kPat[f % 5];
      const double span = 0.0 + pairwiseSum(spans.data(), p.latT);
      t = static_cast<double>(L) + span;
      t = t - 5.0 / 3.0;
    }
    for (int32_t y = 0; y < gh; ++y)
      for (int32_t x = 0; x < gw; ++x) {
        double *r = &pos[size_t(condStart + int32_t(k) * rowsPerFrame + y * gw + x) * 3];
        r[0] = t;
        r[1] = hg[y];
        r[2] = wg[x];
      }
  }
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
  p.tags.assign(M, H3_TAG_VIDEO);
  for (int32_t i = 0; i < L; ++i) p.tags[i] = textTags ? textTags[i] : uint8_t(H3_TAG_TEXT);
  std::fill(p.tags.begin() + condStart, p.tags.begin() + audioStart, uint8_t(H3_TAG_COND));
  std::fill(p.tags.begin() + audioStart, p.tags.begin() + videoStart, uint8_t(H3_TAG_AUDIO));
}

void Prep::schedule(int32_t steps, bool conditioned, Prepared &p) {
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
  p.cT.clear();
  if (conditioned) {  // _row_timestep_plan: conditioning rows at max(t, KEYFRAME_NOISE_AUG) in Python floats, as fp32
    for (float t : p.vT) p.cT.push_back(static_cast<float>(std::max(static_cast<double>(t), 0.999)));
    distinct.insert(p.cT.begin(), p.cT.end());
  }
  p.table.assign(distinct.begin(), distinct.end());
  auto index = [&](const std::vector<float> &t) {
    std::vector<int64_t> idx(t.size());
    for (size_t i = 0; i < t.size(); ++i)
      idx[i] = std::lower_bound(p.table.begin(), p.table.end(), t[i]) - p.table.begin();
    return idx;
  };
  p.tv = index(p.vT);
  p.ta = index(p.aT);
  p.tc = index(p.cT);
}

void Prep::drawBits(uint64_t seed, const std::vector<size_t> &counts, std::vector<std::vector<uint32_t>> &out) {
  // mx.random.seed(seed): key (seed >> 32, seed); every draw takes KeySequence::next(): split(key) -> key = [0],
  // subkey = [1]
  uint32_t k0 = static_cast<uint32_t>(seed >> 32), k1 = static_cast<uint32_t>(seed);
  out.assign(counts.size(), {});
  for (size_t d = 0; d < counts.size(); ++d) {
    uint32_t key[4];
    randomBits(k0, k1, 4, key);
    k0 = key[0];
    k1 = key[1];
    out[d].resize(counts[d]);
    randomBits(key[2], key[3], counts[d], out[d].data());
  }
}

void Prep::noiseBits(uint64_t seed, size_t videoCount, size_t audioCount, std::vector<uint32_t> &video,
                     std::vector<uint32_t> &audio) {
  std::vector<std::vector<uint32_t>> bits;
  drawBits(seed, {videoCount, audioCount}, bits);
  video = std::move(bits[0]);
  audio = std::move(bits[1]);
}

void Prep::run(metal::MetalBackend &b, const std::string &weights, Variant variant, const ops::VideoGeometry &g,
               const PrepInputs &in, Prepared &p) {
  const auto t0 = AwakeClock::now();
  if (!in.hidden || in.textRows < 2) throw std::invalid_argument("prep: text hidden states need >= 2 rows");
  const int32_t L = in.textRows;
  if (in.textTags)
    for (int32_t i = 0; i < L; ++i)
      if (in.textTags[i] != H3_TAG_TEXT && in.textTags[i] != H3_TAG_VIDEO)
        throw std::invalid_argument("prep: text row " + std::to_string(i) + " has tag " +
                                    std::to_string(in.textTags[i]) + " (1 text, 0 vision block)");
  const int32_t steps = in.steps > 0 ? in.steps : variantSteps(variant);
  std::vector<int32_t> anchors;
  if (in.firstFrame) anchors.push_back(0);
  if (in.lastFrame) anchors.push_back(1);
  const int32_t K = static_cast<int32_t>(anchors.size());
  layout(g, L, in.textTags, anchors, p);
  schedule(steps, K > 0, p);
  p.layoutS = since(t0);
  if (K) {  // keyframe moments: the video VAE encoder (released before the text path allocates)
    const auto te = AwakeClock::now();
    const size_t frameBytes = size_t(g.height) * g.width * 3;
    std::vector<uint8_t> rgb;
    for (const uint8_t *f : {in.firstFrame, in.lastFrame})
      if (f) rgb.insert(rgb.end(), f, f + frameBytes);
    ops::VaeEncodeStats es;
    ops::VaeEncoder(b, loadVaeEncoder(b, weights + "/vae")).encodeRgb(rgb.data(), K, g.height, g.width, p.moments, es);
    if (es.nonfinite)
      throw metal::MetalBackendError("prep: " + std::to_string(es.nonfinite) + " non-finite keyframe moments");
    p.encodeS = since(te);
  }
  const std::string ditDir = weights + (variant == Variant::Turbo ? "/dit-turbo" : "/dit-default");
  const SafetensorsCheckpoint dit(ditDir);
  ops::PrepOps op;
  const int32_t T = static_cast<int32_t>(p.table.size()), M = p.rows;
  Buffers buf{b, {}};
  // ---- one command: RoPE, text path, noise
  const auto t1 = AwakeClock::now();
  buf.make("inv_freq", {16}, 4);
  buf.make("pos", {M, 3}, 4);
  buf.make("cos", {M, 48}, 4);
  buf.make("sin", {M, 48}, 4);
  std::memcpy(buf.contents("pos"), p.pos.data(), p.pos.size() * 4);
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
  buf.make("noise_v", {1, H3_VAE_LATENT, p.latT, p.latH, p.latW}, 4);
  buf.make("noise_a", {p.audioRows, 32}, 4);
  const size_t condCount = size_t(p.condRows) * 96, epsCount = size_t(p.latH) * p.latW * H3_VAE_LATENT;
  if (K) {
    // keyframe posterior noise: mx.random.seed(42), one draw per keyframe ([1][1][H][W][24], channels last); then the
    // request's draws in pipeline order: keyframe noise, video latents, audio rows
    std::vector<std::vector<uint32_t>> kbits, rbits;
    drawBits(42, std::vector<size_t>(K, epsCount), kbits);
    drawBits(in.seed, {condCount, videoCount, audioCount}, rbits);
    buf.make("moments", {K, p.latH, p.latW, 48}, 4);
    std::memcpy(buf.contents("moments"), p.moments.data(), p.moments.size() * 4);
    buf.make("bits_k", {int64_t(K * epsCount)}, 4);
    for (int32_t k = 0; k < K; ++k)
      std::memcpy(static_cast<uint32_t *>(buf.contents("bits_k")) + k * epsCount, kbits[k].data(), epsCount * 4);
    buf.make("eps", {K, p.latH, p.latW, H3_VAE_LATENT}, 4);
    buf.make("latent_stats", {48}, 4);
    for (int c = 0; c < 24; ++c) {
      static_cast<float *>(buf.contents("latent_stats"))[c] = static_cast<float>(kVaeLatentMean[c]);
      static_cast<float *>(buf.contents("latent_stats"))[24 + c] = static_cast<float>(kVaeLatentStd[c]);
    }
    buf.make("cond_clean", {p.condRows, 96}, 4);
    buf.make("bits_c", {int64_t(condCount)}, 4);
    std::memcpy(buf.contents("bits_c"), rbits[0].data(), condCount * 4);
    buf.make("noise_c", {p.condRows, 96}, 4);
    buf.make("cond", {p.condRows, 96}, 4);
    vbits = std::move(rbits[1]);
    abits = std::move(rbits[2]);
  } else {
    noiseBits(in.seed, videoCount, audioCount, vbits, abits);
  }
  buf.make("bits_v", {int64_t(videoCount)}, 4);
  buf.make("bits_a", {int64_t(audioCount)}, 4);
  std::memcpy(buf.contents("bits_v"), vbits.data(), videoCount * 4);
  std::memcpy(buf.contents("bits_a"), abits.data(), audioCount * 4);

  auto weight = [&](const std::string &name, const char *dtype, const std::vector<int64_t> &shape) {
    return BufferView{resident(b, dit, name, dtype, shape)};
  };
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
    if (K) {
      op.normal(e, buf["bits_k"], buf["eps"], int(K * epsCount));
      op.keyframeLatents(e, buf["moments"], buf["eps"], buf["latent_stats"], buf["cond_clean"], K, p.latH, p.latW);
      op.normal(e, buf["bits_c"], buf["noise_c"], int(condCount));
      // scale_noise(rows, KEYFRAME_NOISE_AUG, noise): float(t32) and float(1 - t32) in fp32
      const float t32 = 0.999f, omt = 1.0f - t32;
      op.condNoise(e, buf["cond_clean"], buf["noise_c"], buf["cond"], int(condCount), t32, omt);
    }
    video::run(b, e);
  }
  if (K) {
    p.condClean.assign(buf.host<float>("cond_clean"), buf.host<float>("cond_clean") + condCount);
    p.cond.assign(buf.host<float>("cond"), buf.host<float>("cond") + condCount);
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
  // ---- adaLN tables: the converted ones of this schedule (tabulateAdaLn), gathered per step and modality
  const auto t2 = AwakeClock::now();
  const bool keyframes = K > 0;
  const SourceTensor &stored = tensor(dit, adalnName(keyframes, "table"), "F32");
  std::vector<float> table(stored.bytes / 4);
  read(stored, table.data());
  if (table.size() != p.table.size() || std::memcmp(table.data(), p.table.data(), size_t(T) * 4) != 0)
    throw std::invalid_argument("prep: " + ditDir + " holds the adaLN tables of the " +
                                std::to_string(variantSteps(variant)) + "-step schedule, not of " +
                                std::to_string(steps) + " steps");
  const int32_t slots = p.adaModalities();  // video, text, audio (+ conditioning video at max(t, 0.999))
  constexpr size_t kTag = size_t(H3_ADA_ROWS) * kD;
  p.adaBlocks.assign(size_t(steps) * H3_DIT_BLOCKS * slots * kTag, 0);
  p.adaFinal.assign(size_t(steps) * H3_MODALITIES * kFinalOut, 0);
  p.finalLora = dit.find("norm_out.linear.lora_A") != nullptr;
  std::vector<uint16_t> tab(size_t(T) * kAdaOut);
  {  // final layer
    const SourceTensor &final = tensor(dit, adalnName(keyframes, "final"), "BF16", {T, kFinalOut});
    read(final, tab.data());
    p.adalnReadGb += final.bytes / 1e9;
    for (int32_t s = 0; s < steps; ++s)
      for (int m = 0; m < H3_MODALITIES; ++m) {
        const int64_t t = m < 2 ? p.tv[s] : p.ta[s];
        std::memcpy(&p.adaFinal[(size_t(s) * H3_MODALITIES + m) * kFinalOut], &tab[size_t(t) * kFinalOut], kFinalOut * 2);
      }
  }
  const SourceTensor &blocks = tensor(dit, adalnName(keyframes, "blocks"), "BF16", {H3_DIT_BLOCKS, T, kAdaOut});
  const uint64_t blockBytes = uint64_t(T) * kAdaOut * 2;
  for (int i = 0; i < H3_DIT_BLOCKS; ++i) {
    if (dit.find("blocks." + std::to_string(i) + ".adaln_proj.lora_A")) ++p.adalnLoraBlocks;
    blocks.read(uint64_t(i) * blockBytes, {reinterpret_cast<uint8_t *>(tab.data()), blockBytes});
    p.adalnReadGb += blockBytes / 1e9;
    for (int32_t sp = 0; sp < steps; ++sp)
      for (int m = 0; m < slots; ++m) {
        const int64_t t = m == H3_TAG_COND ? p.tc[sp] : (m < 2 ? p.tv[sp] : p.ta[sp]);
        const int src = m == H3_TAG_COND ? H3_TAG_VIDEO : m;  // modality slice of the adaLN output
        std::memcpy(&p.adaBlocks[((size_t(sp) * H3_DIT_BLOCKS + i) * slots + m) * kTag],
                    &tab[size_t(t) * kAdaOut + src * kTag], kTag * 2);
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
  video::run(b, e);
  const float *cs = buf.host<float>("cos"), *sn = buf.host<float>("sin");
  p.rope.resize(size_t(M) * 96);
  for (size_t r = 0; r < size_t(M); ++r)
    for (int c = 0; c < 48; ++c) {
      p.rope[r * 96 + c] = bf16Bits(cs[r * 48 + c]);
      p.rope[r * 96 + 48 + c] = bf16Bits(sn[r * 48 + c]);
    }
}

} // namespace splash::video::model
