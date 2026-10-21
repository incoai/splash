#include "video/h3/model/DitWeights.hpp"

#include "metal/abi/video/Layout.h"
#include "video/h3/model/Synthetic.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <tuple>

namespace h3::model {
namespace {

constexpr int64_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER, QKV = H3_DIT_QKV, G = H3_DIT_GATES;
constexpr int64_t HEADS = H3_DIT_HEADS, HD = H3_DIT_HEAD_DIM;

// bf16 or fp32 source tensor -> fp32 values
std::vector<float> readFloat(const SafetensorsCheckpoint &checkpoint, const std::string &name,
                             const std::vector<int64_t> &shape) {
  const bool bf16 = checkpoint.require(name).dtype == "BF16";
  const SourceTensor &source = tensor(checkpoint, name, bf16 ? "BF16" : "F32", shape);
  std::vector<float> out(source.bytes / (bf16 ? 2 : 4));
  if (!bf16) {
    read(source, out.data());
    return out;
  }
  std::vector<uint16_t> bits(out.size());
  read(source, bits.data());
  for (size_t i = 0; i < out.size(); ++i) {
    const uint32_t u = static_cast<uint32_t>(bits[i]) << 16;
    std::memcpy(&out[i], &u, 4);
  }
  return out;
}

} // namespace

std::unique_ptr<DitWeights> DitWeights::load(metal::MetalBackend &backend, const std::string &dir,
                                             const std::set<int> &qkfixBlocks, bool streamed) {
  auto w = std::unique_ptr<DitWeights>(new DitWeights());
  w->checkpoint_ = std::make_unique<SafetensorsCheckpoint>(dir);
  w->blocks_.resize(H3_DIT_BLOCKS);
  for (int b = 0; b < H3_DIT_BLOCKS; ++b) w->loadBlock(backend, b, streamed);
  for (int b : qkfixBlocks) {
    if (!w->checkpoint_->find("blocks." + std::to_string(b) + ".qkfix.wqkv"))
      throw std::invalid_argument("QKFIX block " + std::to_string(b) + " has no weights in " + dir);
    w->loadQkFix(backend, b, streamed);
  }
  w->loadIo(backend);
  return w;
}

metal::MetalBuffer DitWeights::resident(metal::MetalBackend &backend, const std::string &name, const char *dtype,
                                        const std::vector<int64_t> &shape) {
  const SourceTensor &source = tensor(*checkpoint_, name, dtype, shape);
  metal::MetalBuffer buffer = allocate(backend, source.bytes, name);
  read(source, buffer.contents());
  residentBytes_ += buffer.sizeBytes();
  return buffer;
}

metal::MetalBuffer DitWeights::resident(metal::MetalBackend &backend, const std::vector<float> &values,
                                        const char *what) {
  metal::MetalBuffer buffer = upload(backend, values.data(), values.size() * 4, what);
  residentBytes_ += buffer.sizeBytes();
  return buffer;
}

void DitWeights::loadBlock(metal::MetalBackend &backend, int b, bool streamed) {
  DitBlockWeights &w = blocks_[b];
  const std::string p = "blocks." + std::to_string(b) + ".";
  // int8 matrices: (name, rows, columns, stream tensor, buffer); qkv and gates share the qkvg buffer
  struct Matrix {
    const char *name;
    int64_t rows, cols;
    int slot;
    uint64_t slotOffset;
    metal::MetalBuffer *buffer;
  };
  const Matrix matrices[] = {
      {"attn.qkv.qweight", QKV, D, 0, 0, &w.qkvg},
      {"attn.gates.qweight", G, D, 0, uint64_t(QKV * D), &w.qkvg},
      {"attn.out.qweight", D, C, 1, 0, &w.out},
      {"mlp.fc1.qweight", 2 * F, D, 2, 0, &w.fc1},
      {"mlp.fc2.qweight", D, F, 3, 0, &w.fc2},
      {"attn.out_linear.qweight", D, C, 4, 0, &w.outLinear},
  };
  for (const Matrix &m : matrices) {
    const SourceTensor &source = tensor(*checkpoint_, p + m.name, "I8", {m.rows, m.cols});
    if (streamed) {
      streamed_.push_back({b, m.slot, m.slotOffset, &source, m.name});
      continue;
    }
    if (!*m.buffer) {
      *m.buffer = allocate(backend, m.buffer == &w.qkvg ? uint64_t((QKV + G) * D) : source.bytes, m.name);
      residentBytes_ += m.buffer->sizeBytes();
    }
    read(source, static_cast<uint8_t *>(m.buffer->contents()) + m.slotOffset);
  }
  w.qkvgScale = allocate(backend, (QKV + G) * 4, "qkvg scale");
  residentBytes_ += w.qkvgScale.sizeBytes();
  read(tensor(*checkpoint_, p + "attn.qkv.wscale", "F32", {QKV}), w.qkvgScale.contents());
  read(tensor(*checkpoint_, p + "attn.gates.wscale", "F32", {G}),
       static_cast<uint8_t *>(w.qkvgScale.contents()) + QKV * 4);
  w.outScale = resident(backend, p + "attn.out.wscale", "F32", {D});
  w.fc1Scale = resident(backend, p + "mlp.fc1.wscale", "F32", {2 * F});
  w.fc2Scale = resident(backend, p + "mlp.fc2.wscale", "F32", {D});
  w.outLinearScale = resident(backend, p + "attn.out_linear.wscale", "F32", {D});
  w.norm1 = resident(backend, p + "norm1", "BF16", {D});
  w.norm2 = resident(backend, p + "norm2", "BF16", {D});
  w.qNorm = resident(backend, p + "attn.q_norm", "BF16", {HD});
  w.kNorm = resident(backend, p + "attn.k_norm", "BF16", {HD});
  w.sgBias = resident(backend, p + "vdn.sg_bias", "F32", {HEADS});
  w.ogUp = resident(backend, p + "vdn.og_up", "F16", {C, HD});
  w.ogBias = resident(backend, p + "vdn.og_bias", "F32", {C});
  w.alphaDown = resident(backend, p + "vdn.alpha_down", "F32", {HD, D});
  w.alphaUp = resident(backend, p + "vdn.alpha_up", "F32", {C, HD});
  w.dtBias = resident(backend, p + "vdn.dt_bias", "F32", {C});
  w.aLog = resident(backend, p + "vdn.A_log", "F32", {HEADS});
  w.kSpatial = resident(backend, p + "vdn.k_sp", "F16", {25, C});
  w.kTemporal = resident(backend, p + "vdn.k_tm", "F16", {5, C});
  w.vSpatial = resident(backend, p + "vdn.v_sp", "F16", {25, C});
  w.vTemporal = resident(backend, p + "vdn.v_tm", "F16", {5, C});
  w.branchNorm = resident(backend, p + "vdn.norm", "F32", {HD});
}

void DitWeights::loadQkFix(metal::MetalBackend &backend, int b, bool streamed) {
  const std::string p = "blocks." + std::to_string(b) + ".qkfix.";
  QkFixWeights &q = qkfix_[b];
  const SourceTensor &weights = tensor(*checkpoint_, p + "wqkv", "F16", {QKV, D});
  if (streamed) streamed_.push_back({b, -1, 0, &weights, "qkfix"});
  else q.wqkv = resident(backend, p + "wqkv", "F16", {QKV, D});
  q.splitDims = resident(backend, p + "split", "I32", {4});
}

void DitWeights::loadIo(metal::MetalBackend &backend) {
  const SafetensorsCheckpoint &c = *checkpoint_;
  const int64_t KV = H3_DIT_VIDEO_CHANNELS, KA = H3_DIT_AUDIO_CHANNELS;
  // patch projections [5376][K] -> [K][5376] for coalesced reads in dit_io_embed
  for (auto [name, K, buffer] : {std::tuple<const char *, int64_t, metal::MetalBuffer *>{"video_patch_proj", KV, &io_.videoProj},
                                 std::tuple<const char *, int64_t, metal::MetalBuffer *>{"audio_patch_proj", KA, &io_.audioProj}}) {
    const std::vector<float> v = readFloat(c, std::string(name) + ".weight", {D, K});
    std::vector<float> t(static_cast<size_t>(K * D));
    for (int64_t n = 0; n < D; ++n)
      for (int64_t k = 0; k < K; ++k) t[k * D + n] = v[n * K + k];
    *buffer = resident(backend, t, name);
  }
  io_.videoProjBias = resident(backend, readFloat(c, "video_patch_proj.bias", {D}), "video_patch_proj.bias");
  io_.audioProjBias = resident(backend, readFloat(c, "audio_patch_proj.bias", {D}), "audio_patch_proj.bias");
  io_.finalNorm = resident(backend, "final_layer.norm.weight", "BF16", {D});
  io_.videoOut = resident(backend, readFloat(c, "final_layer.video_out.weight", {KV, D}), "final_layer.video_out.weight");
  io_.videoOutBias = resident(backend, readFloat(c, "final_layer.video_out.bias", {KV}), "final_layer.video_out.bias");
  io_.audioOut = resident(backend, readFloat(c, "final_layer.audio_out.weight", {KA, D}), "final_layer.audio_out.weight");
  io_.audioOutBias = resident(backend, readFloat(c, "final_layer.audio_out.bias", {KA}), "final_layer.audio_out.bias");
}

std::unique_ptr<DitWeights> DitWeights::synthetic(metal::MetalBackend &backend, const std::set<int> &qkfixBlocks,
                                                  const std::set<int> &blocks) {
  auto w = std::unique_ptr<DitWeights>(new DitWeights());
  w->blocks_.resize(H3_DIT_BLOCKS);
  SyntheticFiller f(backend);
  // int8 values are uniform on [-127, 127] (std 73.3): row scales 1 / (73.3 sqrt(K)) x (1 +- 0.25) give unit gain
  auto rowScale = [](int64_t K) { return static_cast<float>(1.0 / (73.32 * std::sqrt(double(K)))); };
  auto uniform = [](int64_t fanIn) { return static_cast<float>(std::sqrt(3.0 / double(fanIn))); };
  for (int b = 0; b < H3_DIT_BLOCKS; ++b) {
    if (!blocks.empty() && !blocks.count(b)) continue;
    DitBlockWeights &x = w->blocks_[b];
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
      QkFixWeights &q = w->qkfix_[b];
      q.wqkv = f.allocate(size_t(QKV * D), H3_SYN_F16, seed(26), uniform(D), 0.0f, "qkfix wqkv");
      const int32_t split[4] = {0, 48, -1, -1};
      q.splitDims = upload(backend, split, sizeof split, "qkfix split");
    }
  }
  const int64_t KV = H3_DIT_VIDEO_CHANNELS, KA = H3_DIT_AUDIO_CHANNELS;
  auto io = [](int id) { return syntheticSeed(kSyntheticIoGroup, id); };
  DitIoWeights &o = w->io_;
  o.videoProj = f.allocate(size_t(KV * D), H3_SYN_F32, io(0), uniform(KV), 0.0f, "video_patch_proj");
  o.videoProjBias = f.allocate(size_t(D), H3_SYN_F32, io(1), 0.1f, 0.0f, "video_patch_proj.bias");
  o.audioProj = f.allocate(size_t(KA * D), H3_SYN_F32, io(2), uniform(KA), 0.0f, "audio_patch_proj");
  o.audioProjBias = f.allocate(size_t(D), H3_SYN_F32, io(3), 0.1f, 0.0f, "audio_patch_proj.bias");
  o.finalNorm = f.allocate(size_t(D), H3_SYN_BF16, io(4), 0.1f, 1.0f, "final norm");
  o.videoOut = f.allocate(size_t(KV * D), H3_SYN_F32, io(5), uniform(D), 0.0f, "video_out");
  o.videoOutBias = f.allocate(size_t(KV), H3_SYN_F32, io(6), 0.1f, 0.0f, "video_out.bias");
  o.audioOut = f.allocate(size_t(KA * D), H3_SYN_F32, io(7), uniform(D), 0.0f, "audio_out");
  o.audioOutBias = f.allocate(size_t(KA), H3_SYN_F32, io(8), 0.1f, 0.0f, "audio_out.bias");
  w->residentBytes_ = f.bytes();
  f.run();
  return w;
}

} // namespace h3::model
