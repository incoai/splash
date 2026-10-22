#include "video/model/DitWeights.hpp"

#include "metal/abi/video/Layout.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <tuple>

namespace splash::video::model {
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

} // namespace splash::video::model
