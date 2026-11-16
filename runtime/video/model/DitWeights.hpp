#pragma once

// DiT weights converted at installation, the safetensors files of dit-<variant>/ read as one checkpoint:
// one file per block (rotated int8 linears + per-row fp32 scales, of the format QUANT_VIDEO_W8 of metal/abi/QuantFormat.h
// that manifest.json names, norms, linear-branch tensors in kernel layouts), qkfix-blockNN.safetensors (fp16 q / k / v
// of the QKFIX blocks + split dims) and io.safetensors (patch projections and final layer). Two block layouts:
// VDN-H3's (q | k | v | 256 gate rows, out, to_out_linear, fc1, fc2, the linear branch) and FastH3's (format
// kFastH3Format of video/model/FastH3.hpp: q | k | v | 7168 VSA compression-gate rows, out, fc1, fc2).
#include "video/metal/Encoder.hpp"
#include "video/model/WeightFiles.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace splash::video::model {

struct DitBlockWeights {
  // int8 linears; empty when streamed (bound per block from the slot ring)
  metal::MetalBuffer qkvg;  // [21504 + 256][5376]: q | k | v rows (head-major), then the gate rows
  metal::MetalBuffer out, fc1, fc2, outLinear;
  metal::MetalBuffer qkvgScale, outScale, fc1Scale, fc2Scale, outLinearScale;
  metal::MetalBuffer norm1, norm2, qNorm, kNorm;  // bf16
  // linear branch: softmax-gate bias, output gate up (fp16 [7168][128]) + bias, alpha down (rotated) / up, dt bias,
  // A_log, depthwise conv taps (fp16 [25 | 5][7168]), readout norm
  metal::MetalBuffer sgBias, ogUp, ogBias, alphaDown, alphaUp, dtBias, aLog;
  metal::MetalBuffer kSpatial, kTemporal, vSpatial, vTemporal, branchNorm;
};

struct QkFixWeights {
  metal::MetalBuffer wqkv;       // fp16 [21504][5376] q | k | v (not rotated); empty when streamed
  metal::MetalBuffer splitDims;  // int32 [4]
};

struct DitIoWeights {
  metal::MetalBuffer videoProj, videoProjBias;  // fp32 [96][5376] (transposed) + [5376]
  metal::MetalBuffer audioProj, audioProjBias;  // fp32 [32][5376] + [5376]
  metal::MetalBuffer finalNorm;                 // bf16 [5376]
  metal::MetalBuffer videoOut, videoOutBias;    // fp32 [96][5376] + [96]
  metal::MetalBuffer audioOut, audioOutBias;    // fp32 [32][5376] + [32]
};

// A tensor left on disk for the weight stream (video/model/WeightStream.hpp), and where it goes in its slot.
struct StreamedTensor {
  int block = 0;
  int slotTensor = 0;       // WeightStream::Tensor, -1 for the QKFIX weights
  uint64_t slotOffset = 0;  // offset inside the slot tensor (gates follow the qkv rows)
  const SourceTensor *tensor = nullptr;
  std::string name;
};

class DitWeights {
public:
  // The five int8 matrices per block and the QKFIX fp16 weights stay on disk, read by the weight stream from the
  // files this object keeps open. The QKFIX blocks are the ones manifest.json's qkfix.blocks names. Refuses a
  // directory whose manifest.json names another format, or no QKFIX blocks.
  static std::unique_ptr<DitWeights> load(metal::MetalBackend &backend, const std::string &dir);
  // Resident weights made in memory, of `bytes` in all (the native tests' synthetic weights);
  // the QKFIX blocks are qkfix's.
  DitWeights(std::vector<DitBlockWeights> blocks, std::map<int, QkFixWeights> qkfix, DitIoWeights io, size_t bytes)
      : blocks_(std::move(blocks)), qkfix_(std::move(qkfix)), io_(std::move(io)), residentBytes_(bytes) {
    for (const auto &entry : qkfix_) qkfixBlocks_.insert(entry.first);
  }

  [[nodiscard]] const DitBlockWeights &block(int b) const { return blocks_[b]; }
  [[nodiscard]] const QkFixWeights &qkfix(int b) const { return qkfix_.at(b); }
  // The QKFIX blocks, whose Q / K stay fp16.
  [[nodiscard]] const std::set<int> &qkfixBlocks() const { return qkfixBlocks_; }
  [[nodiscard]] const DitIoWeights &io() const { return io_; }
  [[nodiscard]] const std::vector<StreamedTensor> &streamed() const { return streamed_; }
  [[nodiscard]] size_t residentBytes() const { return residentBytes_; }
  // FastH3's block layout (video/model/FastH3.hpp).
  [[nodiscard]] bool fastH3() const { return fastH3_; }

private:
  DitWeights() = default;
  void loadBlock(metal::MetalBackend &backend, int b);
  void loadFastH3Block(metal::MetalBackend &backend, int b);  // video/model/FastH3Weights.mm
  void loadQkFix(metal::MetalBackend &backend, int b);
  void loadIo(metal::MetalBackend &backend);
  // A buffer holding the tensor.
  metal::MetalBuffer resident(metal::MetalBackend &backend, const std::string &name, const char *dtype,
                              const std::vector<int64_t> &shape);
  metal::MetalBuffer resident(metal::MetalBackend &backend, const std::vector<float> &values, const char *what);

  std::unique_ptr<SafetensorsCheckpoint> checkpoint_;  // kept open for the streamed tensors
  std::vector<DitBlockWeights> blocks_;
  std::map<int, QkFixWeights> qkfix_;
  std::set<int> qkfixBlocks_;
  DitIoWeights io_;
  std::vector<StreamedTensor> streamed_;
  size_t residentBytes_ = 0;
  bool fastH3_ = false;
};

} // namespace splash::video::model
