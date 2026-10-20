#pragma once

// Converted VDN-H3 DiT weights (tools/convert_dit.py): manifest.json, one safetensors file per block (rotated int8
// linears + per-row fp32 scales, norms, linear-branch tensors in kernel layouts), qkfix-blockNN.safetensors (fp16
// q / k / v of the QKFIX blocks + split dims) and io.safetensors (patch projections and final layer).
#import <Metal/Metal.h>

#include "video/h3/engine/Status.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "video/h3/model/Safetensors.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace h3::model {

struct DitBlockWeights {
  // int8 linears; nil when streamed (bound per block from the slot ring)
  id<MTLBuffer> qkvg = nil;  // [21504 + 256][5376]: q | k | v rows (head-major), then the gate rows
  id<MTLBuffer> out = nil, fc1 = nil, fc2 = nil, outLinear = nil;
  id<MTLBuffer> qkvgScale = nil, outScale = nil, fc1Scale = nil, fc2Scale = nil, outLinearScale = nil;
  id<MTLBuffer> norm1 = nil, norm2 = nil, qNorm = nil, kNorm = nil;  // bf16
  // linear branch: softmax-gate bias, output gate up (fp16 [7168][128]) + bias, alpha down (rotated) / up, dt bias,
  // A_log, depthwise conv taps (fp16 [25 | 5][7168]), readout norm
  id<MTLBuffer> sgBias = nil, ogUp = nil, ogBias = nil, alphaDown = nil, alphaUp = nil, dtBias = nil, aLog = nil;
  id<MTLBuffer> kSpatial = nil, kTemporal = nil, vSpatial = nil, vTemporal = nil, branchNorm = nil;
};

struct QkFixWeights {
  id<MTLBuffer> wqkv = nil;  // fp16 [21504][5376] q | k | v (not rotated); nil when streamed
  id<MTLBuffer> splitDims = nil;  // int32 [4]
};

struct DitIoWeights {
  id<MTLBuffer> videoProj = nil, videoProjBias = nil;  // fp32 [96][5376] (transposed) + [5376]
  id<MTLBuffer> audioProj = nil, audioProjBias = nil;  // fp32 [32][5376] + [5376]
  id<MTLBuffer> finalNorm = nil;                       // bf16 [5376]
  id<MTLBuffer> videoOut = nil, videoOutBias = nil;    // fp32 [96][5376] + [96]
  id<MTLBuffer> audioOut = nil, audioOutBias = nil;    // fp32 [32][5376] + [32]
};

// A tensor left on disk for the stream image, with where its bytes are in the block file.
struct StreamedTensor {
  int block = 0;
  int slotTensor = 0;  // WeightStream::Tensor
  uint64_t slotOffset = 0;  // offset inside the slot tensor (gates follow the qkv rows)
  const SafetensorsFile *file = nullptr;
  TensorInfo info;
  std::string name;
};

class DitWeights {
public:
  // streamed: the five int8 matrices per block and the QKFIX fp16 weights stay on disk.
  static Status load(metal::MetalBackend &backend, const std::string &dir, const std::set<int> &qkfixBlocks,
                     bool streamed, std::unique_ptr<DitWeights> &out);
  // Deterministic synthetic weights (model/Synthetic.hpp) with the converted shapes, resident, no files: the listed
  // blocks (empty: all 50), the QKFIX blocks among them and the io weights.
  static Status synthetic(metal::MetalBackend &backend, const std::set<int> &qkfixBlocks, const std::set<int> &blocks,
                          std::unique_ptr<DitWeights> &out);

  [[nodiscard]] const DitBlockWeights &block(int b) const { return blocks_[b]; }
  [[nodiscard]] const QkFixWeights &qkfix(int b) const { return qkfix_.at(b); }
  [[nodiscard]] const DitIoWeights &io() const { return io_; }
  [[nodiscard]] const std::vector<StreamedTensor> &streamed() const { return streamed_; }
  [[nodiscard]] const std::set<int> &availableQkFixBlocks() const { return available_; }
  [[nodiscard]] size_t residentBytes() const { return residentBytes_; }

private:
  Status loadBlock(metal::MetalBackend &backend, int b, bool streamed);
  Status loadQkFix(metal::MetalBackend &backend, int b, const std::string &file, bool streamed);
  Status loadIo(metal::MetalBackend &backend);
  Status tensor(metal::MetalBackend &backend, const SafetensorsFile &file, const std::string &name, const char *dtype,
                const std::vector<int64_t> &shape, __strong id<MTLBuffer> &out);

  std::string dir_;
  std::vector<SafetensorsFile> files_;  // kept open for the stream-image check
  std::vector<DitBlockWeights> blocks_;
  std::map<int, QkFixWeights> qkfix_;
  std::map<int, SafetensorsFile> qkfixFiles_;
  std::set<int> available_;
  DitIoWeights io_;
  std::vector<StreamedTensor> streamed_;
  size_t residentBytes_ = 0;
};

} // namespace h3::model
