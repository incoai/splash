#pragma once

// VDN linear branch of every block (kernels/dit/linear_branch.metal): gates, frame decays, key / value features,
// per-frame statistics, inverses, scans, window gather and the gated readout of the inner video rows.

#include "video/h3/engine/Encoder.hpp"
#include "video/h3/model/DitWeights.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/ops/Linear.hpp"

#include <string>
#include <vector>

namespace h3::ops {

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
  LinearBranch(const model::SequenceLayout &layout, const BranchBuffers &buffers) : lay_(layout), x_(buffers) {}

  // W8A8 [beta | softmax gate | pad | og_down] on the QKV input (rows 21504.. of the qkvg weight)
  void gates(Encoder &e, ProducerOut in, BufferView qkvg, BufferView qkvgScale, BufferView sgBias) const;
  void frameMean(Encoder &e, ProducerOut in) const;
  void alpha(Encoder &e, const model::DitBlockWeights &w) const;
  void textFeatures(Encoder &e) const;
  void keyFeatures(Encoder &e, const model::DitBlockWeights &w) const;
  void valueFeatures(Encoder &e, const model::DitBlockWeights &w) const;
  void stats(Encoder &e) const;
  void textStats(Encoder &e) const;
  void inverse(Encoder &e) const;
  void textInverse(Encoder &e) const;
  void scan(Encoder &e) const;
  void gather(Encoder &e) const;
  void readout(Encoder &e, const model::DitBlockWeights &w) const;

private:
  void features(Encoder &e, bool key, BufferView spatial, BufferView temporal) const;

  model::SequenceLayout lay_;
  BranchBuffers x_;
};

} // namespace h3::ops
