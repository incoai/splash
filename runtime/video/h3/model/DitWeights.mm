#include "video/h3/model/DitWeights.hpp"

#include "metal/abi/video/Layout.h"
#include "video/h3/model/Json.hpp"
#include "video/h3/model/Synthetic.hpp"

#include <cmath>
#include <cstring>
#include <tuple>

namespace h3::model {
namespace {

constexpr int64_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER, QKV = H3_DIT_QKV, G = H3_DIT_GATES;
constexpr int64_t HEADS = H3_DIT_HEADS, HD = H3_DIT_HEAD_DIM;

// bf16 or fp32 source tensor -> fp32 values
Status readFloat(const SafetensorsFile &file, const std::string &name, const std::vector<int64_t> &shape,
                 std::vector<float> &out) {
  TensorInfo info;
  H3_TRY(file.find(name, nullptr, shape, info));
  size_t count = 1;
  for (int64_t d : shape) count *= static_cast<size_t>(d);
  out.resize(count);
  if (info.dtype == "F32" && info.bytes == count * 4) return file.read(info, out.data());
  if (info.dtype == "BF16" && info.bytes == count * 2) {
    std::vector<uint16_t> bits(count);
    H3_TRY(file.read(info, bits.data()));
    for (size_t i = 0; i < count; ++i) {
      const uint32_t u = static_cast<uint32_t>(bits[i]) << 16;
      std::memcpy(&out[i], &u, 4);
    }
    return {};
  }
  return error(Code::Io, "%s: %s must be F32 or BF16", file.path().c_str(), name.c_str());
}

} // namespace

Status DitWeights::load(metal::MetalBackend &backend, const std::string &dir, const std::set<int> &qkfixBlocks,
                        bool streamed, std::unique_ptr<DitWeights> &out) {
  auto w = std::unique_ptr<DitWeights>(new DitWeights());
  w->dir_ = dir;
  Json manifest;
  H3_TRY(Json::parseFile(dir + "/manifest.json", manifest));
  const Json *files = manifest.get("files");
  if (!files) return error(Code::Io, "%s/manifest.json: no files", dir.c_str());
  w->files_.reserve(H3_DIT_BLOCKS);
  w->blocks_.resize(H3_DIT_BLOCKS);
  for (int b = 0; b < H3_DIT_BLOCKS; ++b) {
    const Json *file = files->get(std::to_string(b));
    if (!file) return error(Code::Io, "%s/manifest.json: no file for block %d", dir.c_str(), b);
    w->files_.emplace_back();
    H3_TRY(SafetensorsFile::open(dir + "/" + file->string(), w->files_.back()));
    H3_TRY(w->loadBlock(backend, b, streamed));
  }
  if (const Json *qkfix = manifest.get("qkfix"))
    if (const Json *blocks = qkfix->get("blocks"))
      for (const auto &[key, rec] : blocks->object()) w->available_.insert(std::stoi(key));
  for (int b : qkfixBlocks) {
    if (!w->available_.count(b)) return error(Code::Argument, "QKFIX block %d has no weights in %s", b, dir.c_str());
    const Json *rec = manifest.get("qkfix")->get("blocks")->get(std::to_string(b));
    const Json *file = rec->get("file");
    if (!file) return error(Code::Io, "%s/manifest.json: QKFIX block %d has no file", dir.c_str(), b);
    H3_TRY(w->loadQkFix(backend, b, file->string(), streamed));
  }
  H3_TRY(w->loadIo(backend));
  out = std::move(w);
  return {};
}

Status DitWeights::tensor(metal::MetalBackend &backend, const SafetensorsFile &file, const std::string &name,
                          const char *dtype, const std::vector<int64_t> &shape, __strong id<MTLBuffer> &out) {
  TensorInfo info;
  H3_TRY(file.find(name, dtype, shape, info));
  H3_TRY(backend.allocate(info.bytes, out, name.c_str()));
  H3_TRY(file.read(info, out.contents));
  residentBytes_ += out.length;
  return {};
}

Status DitWeights::loadBlock(metal::MetalBackend &backend, int b, bool streamed) {
  const SafetensorsFile &file = files_[b];
  DitBlockWeights &w = blocks_[b];
  const std::string p = "blocks." + std::to_string(b) + ".";
  // int8 matrices: (name, rows, columns, stream tensor, buffer); qkv and gates share the qkvg buffer
  struct Matrix {
    const char *name;
    int64_t rows, cols;
    int slot;
    uint64_t slotOffset;
    __strong id<MTLBuffer> *buffer;
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
    TensorInfo info;
    H3_TRY(file.find(p + m.name, "I8", {m.rows, m.cols}, info));
    if (streamed) {
      streamed_.push_back({b, m.slot, m.slotOffset, &file, info, m.name});
      continue;
    }
    if (!*m.buffer) {
      const size_t bytes = m.buffer == &w.qkvg ? size_t((QKV + G) * D) : info.bytes;
      H3_TRY(backend.allocate(bytes, *m.buffer, m.name));
      residentBytes_ += (*m.buffer).length;
    }
    H3_TRY(file.read(info, static_cast<uint8_t *>((*m.buffer).contents) + m.slotOffset));
  }
  H3_TRY(backend.allocate((QKV + G) * 4, w.qkvgScale, "qkvg scale"));
  residentBytes_ += w.qkvgScale.length;
  H3_TRY(file.read(p + "attn.qkv.wscale", "F32", {QKV}, w.qkvgScale.contents, QKV * 4));
  H3_TRY(file.read(p + "attn.gates.wscale", "F32", {G}, static_cast<uint8_t *>(w.qkvgScale.contents) + QKV * 4, G * 4));
  H3_TRY(tensor(backend, file, p + "attn.out.wscale", "F32", {D}, w.outScale));
  H3_TRY(tensor(backend, file, p + "mlp.fc1.wscale", "F32", {2 * F}, w.fc1Scale));
  H3_TRY(tensor(backend, file, p + "mlp.fc2.wscale", "F32", {D}, w.fc2Scale));
  H3_TRY(tensor(backend, file, p + "attn.out_linear.wscale", "F32", {D}, w.outLinearScale));
  H3_TRY(tensor(backend, file, p + "norm1", "BF16", {D}, w.norm1));
  H3_TRY(tensor(backend, file, p + "norm2", "BF16", {D}, w.norm2));
  H3_TRY(tensor(backend, file, p + "attn.q_norm", "BF16", {HD}, w.qNorm));
  H3_TRY(tensor(backend, file, p + "attn.k_norm", "BF16", {HD}, w.kNorm));
  H3_TRY(tensor(backend, file, p + "vdn.sg_bias", "F32", {HEADS}, w.sgBias));
  H3_TRY(tensor(backend, file, p + "vdn.og_up", "F16", {C, HD}, w.ogUp));
  H3_TRY(tensor(backend, file, p + "vdn.og_bias", "F32", {C}, w.ogBias));
  H3_TRY(tensor(backend, file, p + "vdn.alpha_down", "F32", {HD, D}, w.alphaDown));
  H3_TRY(tensor(backend, file, p + "vdn.alpha_up", "F32", {C, HD}, w.alphaUp));
  H3_TRY(tensor(backend, file, p + "vdn.dt_bias", "F32", {C}, w.dtBias));
  H3_TRY(tensor(backend, file, p + "vdn.A_log", "F32", {HEADS}, w.aLog));
  H3_TRY(tensor(backend, file, p + "vdn.k_sp", "F16", {25, C}, w.kSpatial));
  H3_TRY(tensor(backend, file, p + "vdn.k_tm", "F16", {5, C}, w.kTemporal));
  H3_TRY(tensor(backend, file, p + "vdn.v_sp", "F16", {25, C}, w.vSpatial));
  H3_TRY(tensor(backend, file, p + "vdn.v_tm", "F16", {5, C}, w.vTemporal));
  H3_TRY(tensor(backend, file, p + "vdn.norm", "F32", {HD}, w.branchNorm));
  return {};
}

Status DitWeights::loadQkFix(metal::MetalBackend &backend, int b, const std::string &name, bool streamed) {
  SafetensorsFile &file = qkfixFiles_[b];
  H3_TRY(SafetensorsFile::open(dir_ + "/" + name, file));
  const std::string p = "blocks." + std::to_string(b) + ".qkfix.";
  QkFixWeights &q = qkfix_[b];
  TensorInfo info;
  H3_TRY(file.find(p + "wqkv", "F16", {QKV, D}, info));
  if (streamed) streamed_.push_back({b, -1, 0, &file, info, "qkfix"});
  else H3_TRY(tensor(backend, file, p + "wqkv", "F16", {QKV, D}, q.wqkv));
  H3_TRY(tensor(backend, file, p + "split", "I32", {4}, q.splitDims));
  return {};
}

Status DitWeights::loadIo(metal::MetalBackend &backend) {
  SafetensorsFile file;
  H3_TRY(SafetensorsFile::open(dir_ + "/io.safetensors", file));
  const int64_t KV = H3_DIT_VIDEO_CHANNELS, KA = H3_DIT_AUDIO_CHANNELS;
  std::vector<float> v;
  auto put = [&](const std::vector<float> &values, __strong id<MTLBuffer> &out, const char *what) -> Status {
    H3_TRY(backend.upload(values.data(), values.size() * 4, out, what));
    residentBytes_ += out.length;
    return {};
  };
  // patch projections [5376][K] -> [K][5376] for coalesced reads in dit_io_embed
  for (auto [name, K, buffer] : {std::tuple<const char *, int64_t, __strong id<MTLBuffer> *>{"video_patch_proj", KV, &io_.videoProj},
                                 std::tuple<const char *, int64_t, __strong id<MTLBuffer> *>{"audio_patch_proj", KA, &io_.audioProj}}) {
    H3_TRY(readFloat(file, std::string(name) + ".weight", {D, K}, v));
    std::vector<float> t(static_cast<size_t>(K * D));
    for (int64_t n = 0; n < D; ++n)
      for (int64_t k = 0; k < K; ++k) t[k * D + n] = v[n * K + k];
    H3_TRY(put(t, *buffer, name));
  }
  H3_TRY(readFloat(file, "video_patch_proj.bias", {D}, v));
  H3_TRY(put(v, io_.videoProjBias, "video_patch_proj.bias"));
  H3_TRY(readFloat(file, "audio_patch_proj.bias", {D}, v));
  H3_TRY(put(v, io_.audioProjBias, "audio_patch_proj.bias"));
  H3_TRY(tensor(backend, file, "final_layer.norm.weight", "BF16", {D}, io_.finalNorm));
  H3_TRY(readFloat(file, "final_layer.video_out.weight", {KV, D}, v));
  H3_TRY(put(v, io_.videoOut, "final_layer.video_out.weight"));
  H3_TRY(readFloat(file, "final_layer.video_out.bias", {KV}, v));
  H3_TRY(put(v, io_.videoOutBias, "final_layer.video_out.bias"));
  H3_TRY(readFloat(file, "final_layer.audio_out.weight", {KA, D}, v));
  H3_TRY(put(v, io_.audioOut, "final_layer.audio_out.weight"));
  H3_TRY(readFloat(file, "final_layer.audio_out.bias", {KA}, v));
  H3_TRY(put(v, io_.audioOutBias, "final_layer.audio_out.bias"));
  return {};
}

Status DitWeights::synthetic(metal::MetalBackend &backend, const std::set<int> &qkfixBlocks, const std::set<int> &blocks,
                             std::unique_ptr<DitWeights> &out) {
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
    H3_TRY(f.allocate(size_t((QKV + G) * D), H3_SYN_I8, seed(0), 0, 0, x.qkvg, "synthetic qkvg"));
    H3_TRY(f.allocate(size_t(QKV + G), H3_SYN_F32, seed(1), 0.25f * rowScale(D), rowScale(D), x.qkvgScale, "scale"));
    H3_TRY(f.allocate(size_t(D * C), H3_SYN_I8, seed(2), 0, 0, x.out, "synthetic out"));
    H3_TRY(f.allocate(size_t(D), H3_SYN_F32, seed(3), 0.25f * rowScale(C), rowScale(C), x.outScale, "scale"));
    H3_TRY(f.allocate(size_t(2 * F * D), H3_SYN_I8, seed(4), 0, 0, x.fc1, "synthetic fc1"));
    H3_TRY(f.allocate(size_t(2 * F), H3_SYN_F32, seed(5), 0.25f * rowScale(D), rowScale(D), x.fc1Scale, "scale"));
    H3_TRY(f.allocate(size_t(D * F), H3_SYN_I8, seed(6), 0, 0, x.fc2, "synthetic fc2"));
    H3_TRY(f.allocate(size_t(D), H3_SYN_F32, seed(7), 0.25f * rowScale(F), rowScale(F), x.fc2Scale, "scale"));
    H3_TRY(f.allocate(size_t(D * C), H3_SYN_I8, seed(8), 0, 0, x.outLinear, "synthetic out_linear"));
    H3_TRY(f.allocate(size_t(D), H3_SYN_F32, seed(9), 0.25f * rowScale(C), rowScale(C), x.outLinearScale, "scale"));
    H3_TRY(f.allocate(size_t(D), H3_SYN_BF16, seed(10), 0.1f, 1.0f, x.norm1, "norm1"));
    H3_TRY(f.allocate(size_t(D), H3_SYN_BF16, seed(11), 0.1f, 1.0f, x.norm2, "norm2"));
    H3_TRY(f.allocate(size_t(HD), H3_SYN_BF16, seed(12), 0.1f, 1.0f, x.qNorm, "q_norm"));
    H3_TRY(f.allocate(size_t(HD), H3_SYN_BF16, seed(13), 0.1f, 1.0f, x.kNorm, "k_norm"));
    H3_TRY(f.allocate(size_t(HEADS), H3_SYN_F32, seed(14), 0.5f, 0.0f, x.sgBias, "sg_bias"));
    H3_TRY(f.allocate(size_t(C * HD), H3_SYN_F16, seed(15), uniform(HD), 0.0f, x.ogUp, "og_up"));
    H3_TRY(f.allocate(size_t(C), H3_SYN_F32, seed(16), 0.1f, 0.0f, x.ogBias, "og_bias"));
    H3_TRY(f.allocate(size_t(HD * D), H3_SYN_F32, seed(17), uniform(D), 0.0f, x.alphaDown, "alpha_down"));
    H3_TRY(f.allocate(size_t(C * HD), H3_SYN_F32, seed(18), uniform(HD), 0.0f, x.alphaUp, "alpha_up"));
    H3_TRY(f.allocate(size_t(C), H3_SYN_F32, seed(19), 0.5f, 0.0f, x.dtBias, "dt_bias"));
    H3_TRY(f.allocate(size_t(HEADS), H3_SYN_F32, seed(20), 0.5f, 0.0f, x.aLog, "A_log"));
    H3_TRY(f.allocate(size_t(25 * C), H3_SYN_F16, seed(21), 0.2f, 0.0f, x.kSpatial, "k_sp"));
    H3_TRY(f.allocate(size_t(5 * C), H3_SYN_F16, seed(22), 0.2f, 0.0f, x.kTemporal, "k_tm"));
    H3_TRY(f.allocate(size_t(25 * C), H3_SYN_F16, seed(23), 0.2f, 0.0f, x.vSpatial, "v_sp"));
    H3_TRY(f.allocate(size_t(5 * C), H3_SYN_F16, seed(24), 0.2f, 0.0f, x.vTemporal, "v_tm"));
    H3_TRY(f.allocate(size_t(HD), H3_SYN_F32, seed(25), 0.1f, 1.0f, x.branchNorm, "branch norm"));
    if (qkfixBlocks.count(b)) {
      QkFixWeights &q = w->qkfix_[b];
      H3_TRY(f.allocate(size_t(QKV * D), H3_SYN_F16, seed(26), uniform(D), 0.0f, q.wqkv, "qkfix wqkv"));
      const int32_t split[4] = {0, 48, -1, -1};
      H3_TRY(backend.upload(split, sizeof split, q.splitDims, "qkfix split"));
      w->available_.insert(b);
    }
  }
  const int64_t KV = H3_DIT_VIDEO_CHANNELS, KA = H3_DIT_AUDIO_CHANNELS;
  auto io = [](int id) { return syntheticSeed(kSyntheticIoGroup, id); };
  DitIoWeights &o = w->io_;
  H3_TRY(f.allocate(size_t(KV * D), H3_SYN_F32, io(0), uniform(KV), 0.0f, o.videoProj, "video_patch_proj"));
  H3_TRY(f.allocate(size_t(D), H3_SYN_F32, io(1), 0.1f, 0.0f, o.videoProjBias, "video_patch_proj.bias"));
  H3_TRY(f.allocate(size_t(KA * D), H3_SYN_F32, io(2), uniform(KA), 0.0f, o.audioProj, "audio_patch_proj"));
  H3_TRY(f.allocate(size_t(D), H3_SYN_F32, io(3), 0.1f, 0.0f, o.audioProjBias, "audio_patch_proj.bias"));
  H3_TRY(f.allocate(size_t(D), H3_SYN_BF16, io(4), 0.1f, 1.0f, o.finalNorm, "final norm"));
  H3_TRY(f.allocate(size_t(KV * D), H3_SYN_F32, io(5), uniform(D), 0.0f, o.videoOut, "video_out"));
  H3_TRY(f.allocate(size_t(KV), H3_SYN_F32, io(6), 0.1f, 0.0f, o.videoOutBias, "video_out.bias"));
  H3_TRY(f.allocate(size_t(KA * D), H3_SYN_F32, io(7), uniform(D), 0.0f, o.audioOut, "audio_out"));
  H3_TRY(f.allocate(size_t(KA), H3_SYN_F32, io(8), 0.1f, 0.0f, o.audioOutBias, "audio_out.bias"));
  w->residentBytes_ = f.bytes();
  H3_TRY(f.run());
  out = std::move(w);
  return {};
}

} // namespace h3::model
