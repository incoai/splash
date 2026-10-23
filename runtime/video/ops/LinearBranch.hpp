#pragma once

// VDN linear branch of every block (kernels/dit/linear_branch.metal): gates, frame decays, key / value features,
// per-frame statistics, inverses, scans, window gather and the gated readout of the inner video rows.

#include "video/metal/Encoder.hpp"
#include "video/ops/Geometry.hpp"
#include "video/ops/Linear.hpp"

#include <string>
#include <vector>

namespace splash::video::ops {

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
  LinearBranch(const SequenceLayout &layout, const BranchBuffers &buffers) : lay_(layout), x_(buffers) {}

  // W8A8 [beta | softmax gate | pad | og_down] on the QKV input (rows 21504.. of the qkvg weight)
  void gates(Encoder &e, ProducerOut in, BufferView qkvg, BufferView qkvgScale, BufferView sgBias) const;
  void frameMean(Encoder &e, ProducerOut in) const;
  // alpha down (rotated, fp32 [128][5376]) / up (fp32 [7168][128]), dt bias, A_log
  void alpha(Encoder &e, BufferView down, BufferView up, BufferView dtBias, BufferView aLog) const;
  void textFeatures(Encoder &e) const;
  // depthwise conv taps, fp16 [25][7168] spatial and [5][7168] temporal
  void keyFeatures(Encoder &e, BufferView spatial, BufferView temporal) const;
  void valueFeatures(Encoder &e, BufferView spatial, BufferView temporal) const;
  void stats(Encoder &e) const;
  void textStats(Encoder &e) const;
  void inverse(Encoder &e) const;
  void textInverse(Encoder &e) const;
  void scan(Encoder &e) const;
  void gather(Encoder &e) const;
  // readout RMSNorm (fp32 [128]), output gate up (fp16 [7168][128]) + bias
  void readout(Encoder &e, BufferView norm, BufferView gateUp, BufferView gateBias) const;

private:
  void features(Encoder &e, bool key, BufferView spatial, BufferView temporal) const;

  SequenceLayout lay_;
  BranchBuffers x_;
};

} // namespace splash::video::ops
