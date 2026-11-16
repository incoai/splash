#include "video/model/Prep.hpp"

#include "AwakeClock.hpp"
#include "metal/abi/video/Layout.h"
#include "video/model/FastH3.hpp"
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

// bf16 rounding as numpy computes it: round to nearest even on the fp32 bits
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

// The schedule of `steps`, a FastH3 DiT's its trained rungs (the DiT's manifest in ditDir; steps 0 or their count).
void ditSchedule(const std::string &ditDir, bool fastH3, int32_t steps, bool conditioned, Prepared &p) {
  if (!fastH3) return Prep::schedule(steps, conditioned, p);
  const FastH3Contract contract = fastH3Contract(ditDir);
  if (steps > 0 && steps != static_cast<int32_t>(contract.rungs.size()))
    throw std::invalid_argument(std::format("prep: {} runs its {} trained rungs, not {} steps",
                                            std::filesystem::path(ditDir).filename().string().substr(4),
                                            contract.rungs.size(), steps));
  Prep::scheduleRungs(contract, conditioned, p);
}

// The tensors of adaln.safetensors, by schedule: text, keyframes, references.
std::string adalnName(const char *schedule, const char *what) { return std::string("adaln.") + schedule + "." + what; }
constexpr const char *kAdalnSchedules[3] = {"text", "keyframes", "references"};

// References: the conditioning schedule and the clean timestep 1.0 of the reference audio rows (tk).
void addCleanTimestep(Prepared &p) {
  if (!std::binary_search(p.table.begin(), p.table.end(), 1.0f))
    p.table.insert(std::upper_bound(p.table.begin(), p.table.end(), 1.0f), 1.0f);
  auto index = [&](const std::vector<float> &t) {
    std::vector<int64_t> idx(t.size());
    for (size_t i = 0; i < t.size(); ++i)
      idx[i] = std::lower_bound(p.table.begin(), p.table.end(), t[i]) - p.table.begin();
    return idx;
  };
  p.tv = index(p.vT);
  p.ta = index(p.aT);
  p.tc = index(p.cT);
  p.tk.assign(p.vT.size(), std::lower_bound(p.table.begin(), p.table.end(), 1.0f) - p.table.begin());
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

// FL2VA's time embedder on the host, for every variant: fp32 values, fp64 accumulation. The official runs keep it an
// fp32 module with exact fp32 GEMMs (diffusers / OpenVDN, FastVideo); the prep kernels' fp32 matmul (MPP matmul2d)
// truncates its operands to a 10-bit mantissa, which moved every adaLN row by ~0.4 % (rel.), the same way in all 50
// blocks and the final layer: a constant per-latent-channel offset of +0.1..+0.2 in vdn-8's step-0 velocity.
void Prep::timestepEmbedding(const std::vector<float> &t, const std::vector<float> &w1, const std::vector<float> &b1,
                             const std::vector<float> &w2, const std::vector<float> &b2, float *st, uint16_t *stBf) {
  const size_t T = t.size();
  std::vector<float> x(static_cast<size_t>(kTembIn)), h(static_cast<size_t>(kTembHidden));
  for (size_t r = 0; r < T; ++r) {
    for (int i = 0; i < kTembIn / 2; ++i) {  // [cos | sin] (flip_sin_to_cos)
      const float a = t[r] * std::exp(-9.210340371976184f * static_cast<float>(i) / 128.0f);
      x[size_t(i)] = std::cos(a);
      x[size_t(kTembIn / 2 + i)] = std::sin(a);
    }
    float *hp = h.data();
    const float *xp = x.data(), *w1p = w1.data(), *b1p = b1.data(), *w2p = w2.data(), *b2p = b2.data();
    dispatch_apply(size_t(kTembHidden), DISPATCH_APPLY_AUTO, ^(size_t j) {
      double acc = double(b1p[j]);
      for (int k = 0; k < kTembIn; ++k) acc += double(xp[k]) * double(w1p[j * kTembIn + size_t(k)]);
      const double v = double(static_cast<float>(acc));
      hp[j] = static_cast<float>(v / (1.0 + std::exp(-v)));
    });
    float *so = st + r * kTembDim;
    uint16_t *bo = stBf + r * kTembDim;
    dispatch_apply(size_t(kTembDim), DISPATCH_APPLY_AUTO, ^(size_t j) {
      double acc = double(b2p[j]);
      for (int k = 0; k < kTembHidden; ++k) acc += double(hp[k]) * double(w2p[j * kTembHidden + size_t(k)]);
      const double v = double(static_cast<float>(acc));
      so[j] = static_cast<float>(v / (1.0 + std::exp(-v)));
      bo[j] = bf16Bits(so[j]);
    });
  }
}

void Prep::tabulateAdaLn(metal::MetalBackend &b, const std::string &ditDir, const std::string &transformerDir,
                         int32_t steps) {
  const SafetensorsCheckpoint dit(ditDir), transformer(transformerDir);
  const bool fastH3 = isFastH3(ditDir);
  // the schedule without keyframes and with them, whose distinct timesteps index the tables
  struct Schedule {
    Prepared p;
    int32_t T = 0;
    Buffers buf;
    std::vector<double> st;
    metal::MetalBuffer out[2];
  };
  Schedule sched[3] = {{{}, 0, {b, {}}, {}, {}}, {{}, 0, {b, {}}, {}, {}}, {{}, 0, {b, {}}, {}, {}}};
  // references' schedule (+ t = 1): FastH3 DiTs converted with their clean rows (an install spec with "references")
  const int nSched = fastH3 && transformer.find("blocks.0.adaln.clean") ? 3 : 2;
  std::vector<SafetensorsWriter::Entry> entries;
  for (int k = 0; k < nSched; ++k) {
    Schedule &s = sched[k];
    ditSchedule(ditDir, fastH3, steps, k >= 1, s.p);
    if (k == 2) addCleanTimestep(s.p);
    s.T = static_cast<int32_t>(s.p.table.size());
    const uint64_t T = uint64_t(s.T);
    const char *name = kAdalnSchedules[k];
    entries.push_back({adalnName(name, "table"), "F32", {s.T}, T * 4});
    entries.push_back({adalnName(name, "final"), "BF16", {s.T, kFinalOut}, T * kFinalOut * 2});
    entries.push_back({adalnName(name, "blocks"), "BF16", {H3_DIT_BLOCKS, s.T, kAdaOut}, H3_DIT_BLOCKS * T * kAdaOut * 2});
  }
  SafetensorsWriter file(ditDir + "/adaln.safetensors", entries);
  auto weight = [&](const std::string &name, const char *dtype, const std::vector<int64_t> &shape) {
    return BufferView{resident(b, dit, name, dtype, shape)};
  };
  const BufferView finW = weight("final_layer.adaln_proj.linear.weight", "BF16", {kFinalOut, kTembDim});
  const BufferView finB = weight("final_layer.adaln_proj.linear.bias", "BF16", {kFinalOut});
  const std::vector<float> w1 = readF32(dit, "time_embedder.proj_in.weight", {kTembHidden, kTembIn});
  const std::vector<float> b1 = readF32(dit, "time_embedder.proj_in.bias", {kTembHidden});
  const std::vector<float> w2 = readF32(dit, "time_embedder.proj_out.weight", {kTembDim, kTembHidden});
  const std::vector<float> b2 = readF32(dit, "time_embedder.proj_out.bias", {kTembDim});
  ops::PrepOps op;
  {  // timestep MLP and the final layer's adaLN base, of each schedule's distinct timesteps
    Encoder e(b);
    for (int k = 0; k < nSched; ++k) {
      Schedule &s = sched[k];
      const int32_t T = s.T;
      Buffers &buf = s.buf;
      buf.make("st", {T, kTembDim}, 4);
      buf.make("st_bf", {T, kTembDim}, 2);
      buf.make("fin", {T, kFinalOut}, 2);
      for (metal::MetalBuffer &out : s.out) out = allocate(b, size_t(T) * kAdaOut * 2, "adaLN output slot");
      timestepEmbedding(s.p.table, w1, b1, w2, b2, static_cast<float *>(buf.contents("st")),
                        static_cast<uint16_t *>(buf.contents("st_bf")));
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
  for (int k = 0; k < nSched; ++k) {
    Schedule &s = sched[k];
    s.st.resize(size_t(s.T) * kTembDim);
    for (size_t i = 0; i < s.st.size(); ++i) s.st[i] = double(s.buf.host<float>("st")[i]);
    applyLora(s.buf.host<uint16_t>("fin"), s.st, finalLora ? finalA.data() : nullptr,
              finalLora ? finalB.data() : nullptr, s.T, kTembDim, kFinalOut, tab);
    file.put(3 * k, 0, s.p.table.data(), uint64_t(s.T) * 4);
    file.put(3 * k + 1, 0, tab.data(), tab.size() * 2);
  }
  if (fastH3) {  // the students' tables: the block files' of the text schedule, a row per other timestep
    const FastH3Contract contract = fastH3Contract(ditDir);
    if (contract.adalnTimesteps != sched[0].p.table)
      throw std::invalid_argument(ditDir + ": the blocks' adaLN tables are not of the schedule's timesteps");
    std::vector<uint16_t> text(size_t(sched[0].T) * kAdaOut), row(kAdaOut);
    for (int i = 0; i < H3_DIT_BLOCKS; ++i) {
      const std::string b = "blocks." + std::to_string(i) + ".adaln.";
      read(tensor(dit, b + "table", "BF16", {sched[0].T, kAdaOut}), text.data());
      file.put(2, uint64_t(i) * text.size() * 2, text.data(), text.size() * 2);
      const std::vector<float> &table = sched[0].p.table;
      for (int k = 1; k < nSched; ++k)
        for (int32_t j = 0; j < sched[k].T; ++j) {
          const float t = sched[k].p.table[size_t(j)];
          const auto at = std::lower_bound(table.begin(), table.end(), t);
          const uint16_t *source = row.data();
          if (at != table.end() && *at == t) source = &text[size_t(at - table.begin()) * kAdaOut];
          else read(tensor(transformer, b + (t == 1.0f ? "clean" : "keyframe"), "BF16", {1, kAdaOut}), row.data());
          file.put(3 * k + 2, (uint64_t(i) * sched[k].T + j) * kAdaOut * 2, source, uint64_t(kAdaOut) * 2);
        }
    }
    file.publish();
    return;
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
    for (int k = 0; k < nSched; ++k) {
      Schedule &s = sched[k];
      op.linear(e, false, s.buf["st_bf"], wslot[slot], bslot[slot], s.out[slot], s.T, kAdaOut, kTembDim);
    }
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
    for (int k = 0; k < nSched; ++k) {
      Schedule &s = sched[k];
      applyLora(static_cast<const uint16_t *>(s.out[slot].contents()), s.st, lora ? A.data() : nullptr,
                lora ? B.data() : nullptr, s.T, kTembDim, kAdaOut, tab);
      file.put(3 * k + 2, uint64_t(i) * tab.size() * 2, tab.data(), tab.size() * 2);
    }
  }
  file.publish();
}

namespace {

// The fields of p every layout fills from its row classes; pos and tags in the engine order.
void fromSequence(const ops::VideoGeometry &g, ops::RowLayout sequence, Prepared &p) {
  p.frames = g.frames;
  p.textRows = sequence.textRows;
  p.latT = g.latentFrames;
  p.latH = g.latentHeight;
  p.latW = g.latentWidth;
  p.nAud = g.audioRows / 2;
  p.audioRows = g.audioRows;
  p.videoRows = g.videoRows();
  p.rows = sequence.rows();
  // the DiT rows in the engine order (ops::RowLayout::engineOrder; text-to-video and FL2VA: the packed order)
  std::vector<double> pos(sequence.pos.size());
  sequence.toEngineOrder(sequence.pos.data(), pos.data(), 3);
  p.pos.resize(pos.size());
  for (size_t i = 0; i < pos.size(); ++i) p.pos[i] = static_cast<float>(pos[i]);
  p.tags.resize(sequence.slots.size());
  sequence.toEngineOrder(sequence.slots.data(), p.tags.data(), 1);
  p.sequence = std::move(sequence);
}

} // namespace

void Prep::layout(const ops::VideoGeometry &g, int32_t L, const uint8_t *textTags, const std::vector<int32_t> &anchors,
                  Prepared &p) {
  std::vector<ops::ReferenceShape> keyframes;
  for (int32_t a : anchors)
    keyframes.push_back({ops::ReferenceShape::Kind::Keyframe, 1, g.latentHeight, g.latentWidth, 0, a == 1 ? 1 : 0});
  fromSequence(g, ops::RowLayout::build(g, L, textTags, keyframes), p);
  p.condRows = p.sequence.condVideoRows;
}

void Prep::referenceRows(const ops::VideoGeometry &g, const PrepInputs &in, Prepared &p) {
  if (in.firstFrame || in.lastFrame)
    throw std::invalid_argument("prep: keyframes with references (keyframe pixels need the GPU encoder)");
  std::vector<ops::ReferenceShape> shapes;
  for (const ReferenceInput &r : in.references) {
    if (r.shape.kind == ops::ReferenceShape::Kind::Keyframe)
      throw std::invalid_argument("prep: a keyframe is not a reference input");
    const bool needsVideo = r.shape.kind != ops::ReferenceShape::Kind::Audio;
    const bool needsAudio = r.shape.kind == ops::ReferenceShape::Kind::Audio || r.shape.audioLatents > 0;
    if ((needsVideo && !r.video) || (needsAudio && !r.audio))
      throw std::invalid_argument("prep: reference " + std::to_string(shapes.size()) + " without its latents");
    shapes.push_back(r.shape);
  }
  fromSequence(g, ops::RowLayout::build(g, in.textRows, in.textTags, shapes), p);
  p.condRows = 0;  // no keyframes
  p.references = true;
  const ops::RowLayout &l = p.sequence;
  constexpr int32_t C = H3_VAE_LATENT, W = H3_DIT_VIDEO_CHANNELS, A = H3_DIT_AUDIO_CHANNELS;
  p.cond.assign(size_t(l.condVideoRows) * W, 0.0f);
  p.condAudio.assign(size_t(l.condAudioRows) * A, 0.0f);
  size_t vRow = 0, aRow = 0;
  for (const ops::RowSegment &s : l.segments) {
    const ops::RowClassInfo &info = ops::rowClass(s.cls);
    if (info.update == ops::RowUpdate::Noised || info.stream == ops::RowStream::Text) continue;
    const ReferenceInput &r = in.references[size_t(s.ref)];
    if (info.stream == ops::RowStream::Video) {
      // patchify_video_latents: [24][F][h][w] -> rows [(f, y / 2, x / 2)][c x 4 + 2 (y % 2) + x % 2]
      const int32_t F = s.frames, h = s.latentHeight, w = s.latentWidth, gw = w / 2, rowsPerFrame = (h / 2) * gw;
      for (int32_t c = 0; c < C; ++c)
        for (int32_t f = 0; f < F; ++f)
          for (int32_t y = 0; y < h; ++y)
            for (int32_t x = 0; x < w; ++x) {
              const size_t row = vRow + size_t(f) * rowsPerFrame + (y / 2) * gw + x / 2;
              p.cond[row * W + c * 4 + (y % 2) * 2 + x % 2] = r.video[((size_t(c) * F + f) * h + y) * w + x];
            }
      vRow += s.rows;
    } else {
      // channel-major rows: channel 0's latents, then channel 1's
      const int32_t T = s.rows / 2;
      for (int32_t ch = 0; ch < 2; ++ch)
        for (int32_t t = 0; t < T; ++t)
          std::memcpy(&p.condAudio[(aRow + size_t(ch) * T + t) * A], &r.audio[(size_t(t) * 2 + ch) * A], A * 4);
      aRow += s.rows;
    }
  }
  if (in.referenceNoise) {
    // scale_noise(rows, KEYFRAME_NOISE_AUG, noise): float(t32) and float(1 - t32) in fp32, as prep_cond_noise
    const float t32 = ops::kConditionTime, omt = 1.0f - t32;
    for (size_t i = 0; i < p.cond.size(); ++i) {
      const float a = t32 * p.cond[i];
      const float b = omt * in.referenceNoise[i];
      p.cond[i] = a + b;
    }
  }
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

void Prep::run(metal::MetalBackend &b, const std::string &weights, const std::string &ditName,
               const ops::VideoGeometry &g, const PrepInputs &in, Prepared &p) {
  const auto t0 = AwakeClock::now();
  if (!in.hidden || in.textRows < 2) throw std::invalid_argument("prep: text hidden states need >= 2 rows");
  const int32_t L = in.textRows;
  const bool refs = !in.references.empty();
  if (in.textTags)
    for (int32_t i = 0; i < L; ++i)
      if (in.textTags[i] != H3_TAG_TEXT && in.textTags[i] != H3_TAG_VIDEO)
        throw std::invalid_argument("prep: text row " + std::to_string(i) + " has tag " +
                                    std::to_string(in.textTags[i]) + " (1 text, 0 vision block)");
  const std::string ditDir = weights + "/" + ditDirectory(ditName);
  std::vector<int32_t> anchors;
  if (in.firstFrame) anchors.push_back(0);
  if (in.lastFrame) anchors.push_back(1);
  const int32_t K = static_cast<int32_t>(anchors.size());
  if (refs) {  // Ref2VA: the references' rows and their clean (or given augmented) latents, on the CPU
    referenceRows(g, in, p);
  } else {
    layout(g, L, in.textTags, anchors, p);
  }
  ditSchedule(ditDir, isFastH3(ditDir), in.steps, K > 0 || refs, p);
  if (refs) addCleanTimestep(p);
  const int32_t steps = p.steps;
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
  {  // the refiner blocks share one set of scratch rows (they run in order; block 1 reads block 0's x2 before it
     // writes x1 and x2): 0.28 MB a text row instead of 0.50, 5.9 GB for Ref2VA's 21,045 text rows at 768p
    const std::string s = "b_";
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
  const int32_t condRows = refs ? p.sequence.condVideoRows : p.condRows;
  const size_t condCount = size_t(condRows) * 96, epsCount = size_t(p.latH) * p.latW * H3_VAE_LATENT;
  // references: their augmentation noise is the request's first draw, as the keyframes' (unless given)
  const bool augment = refs && !in.referenceNoise && condCount;
  if (refs) {
    std::vector<std::vector<uint32_t>> rbits;
    drawBits(in.seed, {condCount, videoCount, audioCount}, rbits);
    if (augment) {
      buf.make("cond_clean", {condRows, 96}, 4);
      std::memcpy(buf.contents("cond_clean"), p.cond.data(), condCount * 4);
      buf.make("bits_c", {int64_t(condCount)}, 4);
      std::memcpy(buf.contents("bits_c"), rbits[0].data(), condCount * 4);
      buf.make("noise_c", {condRows, 96}, 4);
      buf.make("cond", {condRows, 96}, 4);
    }
    vbits = std::move(rbits[1]);
    abits = std::move(rbits[2]);
  } else if (K) {
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
  // Pika's sampler: the fresh noise of every step but the last, from the request's second stream
  const int32_t renoiseSteps = p.renoise ? steps - 1 : 0;
  if (renoiseSteps > 0) {
    std::vector<size_t> counts;
    for (int32_t k = 0; k < renoiseSteps; ++k) {
      counts.push_back(videoCount);
      counts.push_back(audioCount);
    }
    std::vector<std::vector<uint32_t>> rn;
    drawBits(in.seed + p.renoiseSeedOffset, counts, rn);
    buf.make("rn_bits_v", {int64_t(renoiseSteps * videoCount)}, 4);
    buf.make("rn_bits_a", {int64_t(renoiseSteps * audioCount)}, 4);
    buf.make("rn_v", {int64_t(renoiseSteps * videoCount)}, 4);
    buf.make("rn_a", {int64_t(renoiseSteps * audioCount)}, 4);
    for (int32_t k = 0; k < renoiseSteps; ++k) {
      std::memcpy(static_cast<uint32_t *>(buf.contents("rn_bits_v")) + k * videoCount, rn[2 * k].data(), videoCount * 4);
      std::memcpy(static_cast<uint32_t *>(buf.contents("rn_bits_a")) + k * audioCount, rn[2 * k + 1].data(),
                  audioCount * 4);
    }
  }

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
  // the token refiner's runtime LoRA (video/model/Adapter.hpp): per linear y = W x + B (A x), bf16 linears as PEFT runs them
  if (in.adapters.size() > 1) throw std::invalid_argument("prep: one runtime adapter at a time");
  struct RefinerLora {
    BufferView a, b;
    int32_t rows = 0;
  };
  struct RefinerLoraBlock {
    RefinerLora qkv, out, fc1, fc2;
  } rl[kRefinerBlocks];
  bool refinerLora = false;
  if (!in.adapters.empty()) {
    float scale = 1.0f;
    std::vector<Adapter::RefinerBlock> blocks = Adapter::refiner(in.adapters[0], kRefinerBlocks, &scale);
    refinerLora = !blocks.empty();
    for (int blk = 0; refinerLora && blk < kRefinerBlocks; ++blk) {
      auto put = [&](Adapter::RefinerLinear &l, const std::string &what) {
        if (scale != 1.0f)  // PEFT scales B (A x) after B; folded into B (bf16) here
          for (uint16_t &v : l.b) v = bf16Bits(fromBf16(v) * scale);
        return RefinerLora{upload(b, l.a.data(), l.a.size() * 2, what + " A"), upload(b, l.b.data(), l.b.size() * 2,
                                                                                      what + " B"),
                           l.rows};
      };
      const std::string s = "refiner LoRA " + std::to_string(blk) + " ";
      rl[blk] = {put(blocks[blk].qkv, s + "qkv"), put(blocks[blk].out, s + "out"), put(blocks[blk].fc1, s + "fc1"),
                 put(blocks[blk].fc2, s + "fc2")};
    }
    buf.make("lora_t", {L, 3 * 128}, 2);
    buf.make("lora_u", {L, 2 * kFfn}, 2);
  }
  // y += B (A x): bf16 A x, bf16 B (A x), then the bf16 add (y may be the linear's output in place)
  auto addLora = [&](Encoder &e, const RefinerLora &l, BufferView xin, BufferView y, int N, int K) {
    op.linear(e, false, xin, l.a, {}, buf["lora_t"], L, l.rows, K);
    op.linear(e, false, buf["lora_t"], l.b, {}, buf["lora_u"], L, N, l.rows);
    op.add(e, y, buf["lora_u"], y, L * N);
  };
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
      const std::string s = "b_";
      const RefinerBlock &r = rb[blk];
      op.rms(e, buf[x], r.norm1, buf[s + "n1"], L, kD, kD, kD, kEps);
      op.linear(e, false, buf[s + "n1"], r.qkv, {}, buf[s + "qkv"], L, 3 * kInner, kD);
      if (refinerLora) addLora(e, rl[blk].qkv, buf[s + "n1"], buf[s + "qkv"], 3 * kInner, kD);
      op.rms(e, buf[s + "qkv"], r.qNorm, buf[s + "qn"], L * kHeads, kHeadDim, 3 * kHeadDim, kHeadDim, kEps);
      op.rms(e, buf[s + "qkv"] + size_t(kHeadDim) * 2, r.kNorm, buf[s + "kn"], L * kHeads, kHeadDim, 3 * kHeadDim,
             kHeadDim, kEps);
      const ops::PrepOps::AttnStrides st{kHeadDim, kInner, kHeadDim, kInner, 3 * kHeadDim, 3 * kInner, kHeadDim, kInner};
      op.attention(e, buf[s + "qn"], buf[s + "kn"], buf[s + "qkv"] + size_t(2 * kHeadDim) * 2, buf[s + "att"], L,
                   kHeads, attnScale, st);
      op.linear(e, false, buf[s + "att"], r.out, {}, buf[s + "o"], L, kD, kInner);
      if (refinerLora) addLora(e, rl[blk].out, buf[s + "att"], buf[s + "o"], kD, kInner);
      op.add(e, buf[x], buf[s + "o"], buf[s + "x1"], L * kD);
      op.rms(e, buf[s + "x1"], r.norm2, buf[s + "n2"], L, kD, kD, kD, kEps);
      op.linear(e, false, buf[s + "n2"], r.fc1, {}, buf[s + "f1"], L, 2 * kFfn, kD);
      if (refinerLora) addLora(e, rl[blk].fc1, buf[s + "n2"], buf[s + "f1"], 2 * kFfn, kD);
      op.swiglu(e, buf[s + "f1"], buf[s + "act"], L, kFfn);
      op.linear(e, false, buf[s + "act"], r.fc2, {}, buf[s + "m"], L, kD, kFfn);
      if (refinerLora) addLora(e, rl[blk].fc2, buf[s + "act"], buf[s + "m"], kD, kFfn);
      op.add(e, buf[s + "x1"], buf[s + "m"], buf[s + "x2"], L * kD);
      x = s + "x2";
    }
    op.rms(e, buf[x], finalNorm, buf["final"], L, kD, kD, kD, kEps);
    op.normal(e, buf["bits_v"], buf["noise_v"], int(videoCount));
    op.normal(e, buf["bits_a"], buf["noise_a"], int(audioCount));
    if (renoiseSteps > 0) {
      op.normal(e, buf["rn_bits_v"], buf["rn_v"], int(renoiseSteps * videoCount));
      op.normal(e, buf["rn_bits_a"], buf["rn_a"], int(renoiseSteps * audioCount));
    }
    if (K) op.normal(e, buf["bits_k"], buf["eps"], int(K * epsCount));
    if (K) op.keyframeLatents(e, buf["moments"], buf["eps"], buf["latent_stats"], buf["cond_clean"], K, p.latH, p.latW);
    if (K || augment) {
      op.normal(e, buf["bits_c"], buf["noise_c"], int(condCount));
      // scale_noise(rows, KEYFRAME_NOISE_AUG, noise): float(t32) and float(1 - t32) in fp32
      const float t32 = 0.999f, omt = 1.0f - t32;
      op.condNoise(e, buf["cond_clean"], buf["noise_c"], buf["cond"], int(condCount), t32, omt);
    }
    video::run(b, e);
  }
  if (K || augment) {
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
    const int32_t F = p.latT, Hh = p.latH, Ww = p.latW, gh = Hh / 2, gw = Ww / 2;
    auto patchify = [&](const float *v, float *rows) {
      for (int32_t f = 0; f < F; ++f)
        for (int32_t y = 0; y < gh; ++y)
          for (int32_t x = 0; x < gw; ++x)
            for (int32_t c = 0; c < 24; ++c)
              for (int32_t py = 0; py < 2; ++py)
                for (int32_t px = 0; px < 2; ++px)
                  rows[((size_t(f) * gh + y) * gw + x) * 96 + c * 4 + py * 2 + px] =
                      v[((size_t(c) * F + f) * Hh + 2 * y + py) * Ww + 2 * x + px];
    };
    p.videoNoise.resize(size_t(p.videoRows) * 96);
    patchify(buf.host<float>("noise_v"), p.videoNoise.data());
    p.audioNoise.assign(buf.host<float>("noise_a"), buf.host<float>("noise_a") + audioCount);
    if (renoiseSteps > 0) {
      p.renoiseVideo.resize(size_t(renoiseSteps) * p.videoRows * 96);
      for (int32_t k = 0; k < renoiseSteps; ++k)
        patchify(buf.host<float>("rn_v") + size_t(k) * videoCount, p.renoiseVideo.data() + size_t(k) * p.videoRows * 96);
      p.renoiseAudio.assign(buf.host<float>("rn_a"), buf.host<float>("rn_a") + size_t(renoiseSteps) * audioCount);
    }
  }
  // ---- adaLN tables: the converted ones of this schedule (tabulateAdaLn), gathered per step and modality
  const auto t2 = AwakeClock::now();
  const char *schedule = kAdalnSchedules[refs ? 2 : (K > 0 ? 1 : 0)];
  if (refs && !dit.find(adalnName(schedule, "table")))
    throw std::invalid_argument("prep: " + ditDir + " has no adaLN tables of references (install it from a spec "
                                "with \"references\": true)");
  const SourceTensor &stored = tensor(dit, adalnName(schedule, "table"), "F32");
  std::vector<float> table(stored.bytes / 4);
  read(stored, table.data());
  if (table.size() != p.table.size() || std::memcmp(table.data(), p.table.data(), size_t(T) * 4) != 0)
    throw std::invalid_argument("prep: " + ditDir + " holds the adaLN tables of another schedule, not of " +
                                std::to_string(steps) + " steps");
  // video, text, audio (+ conditioning video at max(t, 0.999), + conditioning audio at 1)
  const int32_t slots = p.adaModalities(), finalSlots = p.finalSlots();
  constexpr size_t kTag = size_t(H3_ADA_ROWS) * kD;
  auto timestepOf = [&](int m, int32_t s) {
    return m == ops::kSlotCondAudio ? p.tk[s] : m == H3_TAG_COND ? p.tc[s] : (m < 2 ? p.tv[s] : p.ta[s]);
  };
  p.adaBlocks.assign(size_t(steps) * H3_DIT_BLOCKS * slots * kTag, 0);
  p.adaFinal.assign(size_t(steps) * finalSlots * kFinalOut, 0);
  std::vector<uint16_t> tab(size_t(T) * kAdaOut);
  {  // final layer
    const SourceTensor &final = tensor(dit, adalnName(schedule, "final"), "BF16", {T, kFinalOut});
    read(final, tab.data());
    p.adalnReadGb += final.bytes / 1e9;
    for (int32_t s = 0; s < steps; ++s)
      for (int m = 0; m < finalSlots; ++m)
        std::memcpy(&p.adaFinal[(size_t(s) * finalSlots + m) * kFinalOut], &tab[size_t(timestepOf(m, s)) * kFinalOut],
                    kFinalOut * 2);
  }
  const SourceTensor &blocks = tensor(dit, adalnName(schedule, "blocks"), "BF16", {H3_DIT_BLOCKS, T, kAdaOut});
  const uint64_t blockBytes = uint64_t(T) * kAdaOut * 2;
  for (int i = 0; i < H3_DIT_BLOCKS; ++i) {
    blocks.read(uint64_t(i) * blockBytes, {reinterpret_cast<uint8_t *>(tab.data()), blockBytes});
    p.adalnReadGb += blockBytes / 1e9;
    for (int32_t sp = 0; sp < steps; ++sp)
      for (int m = 0; m < slots; ++m) {
        const int64_t t = timestepOf(m, sp);
        // modality slice of the adaLN output: conditioning video takes the video's, conditioning audio the audio's
        const int src = m == H3_TAG_COND ? H3_TAG_VIDEO : (m == ops::kSlotCondAudio ? H3_TAG_AUDIO : m);
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
