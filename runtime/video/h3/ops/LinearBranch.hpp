#pragma once

// VDN linear branch of every block (kernels/dit/linear_branch.metal): gates, frame decays, key / value features,
// per-frame statistics, inverses, scans, window gather and the gated readout of the inner video rows.
#import <Metal/Metal.h>

#include "video/h3/metal/MetalBackend.hpp"
#include "video/h3/model/DitWeights.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/ops/Linear.hpp"

#include <string>
#include <vector>

namespace h3::ops {

[[nodiscard]] std::vector<std::string> branchKernels();

struct BranchBuffers {
  BufferView rawQ, rawK, v;                 // raw q / k [56][rawRows][128], V [56][Spad][128]
  BufferView beta, softmaxGate, outGateHidden;
  BufferView frameMean, alpha;
  BufferView textKeys, textValues, keyFeatures, valueFeatures;
  BufferView statsA, statsB, inverse, textA, textB, textInverse, pivots, textPivots;
  BufferView scanPre, scanSuf, scanStart, scanScratch;
  BufferView state, stateScale, bounds, readout;
  int32_t paddedRows, rawRows;
};

class LinearBranch {
public:
  LinearBranch(const metal::MetalBackend &backend, const model::SequenceLayout &layout, const BranchBuffers &buffers)
      : b_(backend), lay_(layout), x_(buffers) {}

  // W8A8 [beta | softmax gate | pad | og_down] on the QKV input (rows 21504.. of the qkvg weight)
  void gates(id<MTLComputeCommandEncoder> e, ProducerOut in, BufferView qkvg, BufferView qkvgScale,
             BufferView sgBias) const;
  void frameMean(id<MTLComputeCommandEncoder> e, ProducerOut in) const;
  void alpha(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const;
  void textFeatures(id<MTLComputeCommandEncoder> e) const;
  void keyFeatures(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const;
  void valueFeatures(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const;
  void stats(id<MTLComputeCommandEncoder> e) const;
  void textStats(id<MTLComputeCommandEncoder> e) const;
  void inverse(id<MTLComputeCommandEncoder> e) const;
  void textInverse(id<MTLComputeCommandEncoder> e) const;
  void scan(id<MTLComputeCommandEncoder> e) const;
  void gather(id<MTLComputeCommandEncoder> e) const;
  void readout(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const;

private:
  void features(id<MTLComputeCommandEncoder> e, bool key, BufferView spatial, BufferView temporal) const;

  const metal::MetalBackend &b_;
  model::SequenceLayout lay_;
  BranchBuffers x_;
};

} // namespace h3::ops
