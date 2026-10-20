#include "video/h3/ops/LinearBranch.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/VdnABI.h"

namespace h3::ops {
namespace {
constexpr int32_t D = H3_DIT_HIDDEN, C = H3_DIT_INNER, H = H3_DIT_HEADS, G = H3_DIT_GATES;
constexpr int32_t kReadoutSimdgroups = 2;
}  // namespace

std::vector<std::string> branchKernels() {
  return {"vdn_gates", "vdn_frame_mean",   "vdn_alpha", "vdn_text_features", "vdn_conv_features",
          "vdn_stats", "vdn_inverse", "vdn_scan", "vdn_gather", "vdn_readout"};
}

// Parameter block of the frame / feature / state kernels: inner frames of T tokens on a gh x gw grid.
static BranchParams branchParams(const model::SequenceLayout &lay) {
  BranchParams p{};
  p.H = H;
  p.S = lay.tokensPerFrame;
  p.gh = lay.gridH;
  p.gw = lay.gridW;
  p.Fp = lay.innerFrames();
  p.ldo = C;
  p.eps = device::kBranchEps;
  return p;
}

void LinearBranch::gates(id<MTLComputeCommandEncoder> e, ProducerOut in, BufferView qkvg, BufferView qkvgScale,
                         BufferView sgBias) const {
  const int32_t M = lay_.rows();
  const BranchGateParams p{M, G, D};
  [e setComputePipelineState:b_.pipeline("vdn_gates")];
  metal::bind(e, in.q, 0);
  metal::bind(e, qkvg + size_t(H3_DIT_QKV) * D, 1);
  metal::bind(e, in.scale, 2);
  metal::bind(e, qkvgScale + size_t(H3_DIT_QKV) * 4, 3);
  metal::bind(e, x_.beta, 4);
  metal::bind(e, x_.softmaxGate, 5);
  metal::bind(e, x_.outGateHidden, 6);
  metal::bind(e, sgBias, 7);
  metal::bindBytes(e, p, 8);
  metal::dispatch(e, MTLSizeMake(G / 128, (M + 63) / 64, 1), MTLSizeMake(128, 1, 1));
}

void LinearBranch::frameMean(id<MTLComputeCommandEncoder> e, ProducerOut in) const {
  const BranchFrameParams p{lay_.tokensPerFrame, D, H, lay_.innerFrames(), lay_.innerRow0()};
  [e setComputePipelineState:b_.pipeline("vdn_frame_mean")];
  metal::bind(e, in.q, 0);
  metal::bind(e, in.scale, 1);
  metal::bind(e, x_.frameMean, 2);
  metal::bindBytes(e, p, 3);
  metal::dispatch(e, MTLSizeMake((D / 4 + 255) / 256, lay_.innerFrames(), 1), MTLSizeMake(256, 1, 1));
}

void LinearBranch::alpha(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const {
  const BranchFrameParams p{lay_.tokensPerFrame, D, H, lay_.innerFrames(), lay_.innerRow0()};
  [e setComputePipelineState:b_.pipeline("vdn_alpha")];
  metal::bind(e, x_.frameMean, 0);
  metal::bind(e, w.alphaDown, 1);
  metal::bind(e, w.alphaUp, 2);
  metal::bind(e, w.dtBias, 3);
  metal::bind(e, w.aLog, 4);
  metal::bind(e, x_.alpha, 5);
  metal::bindBytes(e, p, 6);
  metal::dispatch(e, MTLSizeMake(lay_.innerFrames(), 1, 1), MTLSizeMake(1024, 1, 1));
}

// text rows of raw k (L2) and of V: the state start of both scans
void LinearBranch::textFeatures(id<MTLComputeCommandEncoder> e) const {
  for (int kv = 0; kv < 2; ++kv) {
    BranchParams p = branchParams(lay_);
    p.row0 = 0;
    p.l2 = kv == 0;
    p.nrows = lay_.text;
    p.Spad = kv == 0 ? x_.rawRows : x_.paddedRows;
    [e setComputePipelineState:b_.pipeline("vdn_text_features")];
    metal::bind(e, kv == 0 ? x_.rawK : x_.v, 0);
    metal::bind(e, kv == 0 ? x_.textKeys : x_.textValues, 1);
    metal::bind(e, x_.beta, 2);
    metal::bindBytes(e, p, 3);
    metal::dispatch(e, MTLSizeMake((lay_.text + 7) / 8, H, 1), MTLSizeMake(256, 1, 1));
  }
}

void LinearBranch::features(id<MTLComputeCommandEncoder> e, bool key, BufferView spatial, BufferView temporal) const {
  BranchParams p = branchParams(lay_);
  p.l2 = key;
  p.Spad = key ? x_.rawRows : x_.paddedRows;
  p.row0 = key ? lay_.text : lay_.innerRow0();  // raw k is compact: inner frames start after the text rows
  [e setComputePipelineState:b_.pipeline("vdn_conv_features")];
  metal::bind(e, key ? x_.rawK : x_.v, 0);
  metal::bind(e, key ? x_.keyFeatures : x_.valueFeatures, 1);
  metal::bind(e, spatial, 2);
  metal::bind(e, temporal, 3);
  metal::bind(e, x_.beta + size_t(lay_.innerRow0()) * H * 4, 4);
  metal::bindBytes(e, p, 5);
  const int32_t tiles = ((lay_.gridH + 5) / 6) * ((lay_.gridW + 5) / 6);  // 6 x 6 token tiles
  metal::dispatch(e, MTLSizeMake(tiles, H, 1), MTLSizeMake(32 * 3 * 6, 1, 1));
}

void LinearBranch::keyFeatures(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const {
  features(e, true, w.kSpatial, w.kTemporal);
}

void LinearBranch::valueFeatures(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const {
  features(e, false, w.vSpatial, w.vTemporal);
}

void LinearBranch::stats(id<MTLComputeCommandEncoder> e) const {
  const BranchParams p = branchParams(lay_);
  [e setComputePipelineState:b_.pipeline("vdn_stats")];
  metal::bind(e, x_.keyFeatures, 0);
  metal::bind(e, x_.valueFeatures, 1);
  metal::bind(e, x_.statsA, 2);
  metal::bind(e, x_.statsB, 3);
  metal::bindBytes(e, p, 4);
  metal::dispatch(e, MTLSizeMake(4, H, lay_.innerFrames()), MTLSizeMake(128, 1, 1));
}

void LinearBranch::textStats(id<MTLComputeCommandEncoder> e) const {
  BranchParams p = branchParams(lay_);
  p.Fp = 1;
  p.S = lay_.text;
  [e setComputePipelineState:b_.pipeline("vdn_stats")];
  metal::bind(e, x_.textKeys, 0);
  metal::bind(e, x_.textValues, 1);
  metal::bind(e, x_.textA, 2);
  metal::bind(e, x_.textB, 3);
  metal::bindBytes(e, p, 4);
  metal::dispatch(e, MTLSizeMake(4, H, 1), MTLSizeMake(128, 1, 1));
}

void LinearBranch::inverse(id<MTLComputeCommandEncoder> e) const {
  [e setComputePipelineState:b_.pipeline("vdn_inverse")];
  metal::bind(e, x_.statsA, 0);
  metal::bind(e, x_.inverse, 1);
  metal::bind(e, x_.pivots, 2);
  metal::dispatch(e, MTLSizeMake(lay_.innerFrames() * H, 1, 1), MTLSizeMake(1024, 1, 1));
}

void LinearBranch::textInverse(id<MTLComputeCommandEncoder> e) const {
  [e setComputePipelineState:b_.pipeline("vdn_inverse")];
  metal::bind(e, x_.textA, 0);
  metal::bind(e, x_.textInverse, 1);
  metal::bind(e, x_.textPivots, 2);
  metal::dispatch(e, MTLSizeMake(H, 1, 1), MTLSizeMake(1024, 1, 1));
}

void LinearBranch::scan(id<MTLComputeCommandEncoder> e) const {
  const BranchParams p = branchParams(lay_);
  [e setComputePipelineState:b_.pipeline("vdn_scan")];
  metal::bind(e, x_.inverse, 0);
  metal::bind(e, x_.statsB, 1);
  metal::bind(e, x_.alpha, 2);
  metal::bind(e, x_.scanPre, 3);
  metal::bind(e, x_.scanSuf, 4);
  metal::bind(e, x_.textInverse, 5);
  metal::bind(e, x_.textB, 6);
  metal::bind(e, x_.scanStart, 7);
  metal::bind(e, x_.scanScratch, 8);
  metal::bindBytes(e, p, 9);
  metal::dispatch(e, MTLSizeMake(H, 2, 1), MTLSizeMake(1024, 1, 1));
}

void LinearBranch::gather(id<MTLComputeCommandEncoder> e) const {
  const BranchParams p = branchParams(lay_);
  [e setComputePipelineState:b_.pipeline("vdn_gather")];
  metal::bind(e, x_.scanPre, 0);
  metal::bind(e, x_.scanSuf, 1);
  metal::bind(e, x_.alpha, 2);
  metal::bind(e, x_.scanStart, 3);
  metal::bind(e, x_.bounds, 4);
  metal::bind(e, x_.state, 5);
  metal::bind(e, x_.stateScale, 6);
  metal::bindBytes(e, p, 7);
  metal::dispatch(e, MTLSizeMake(lay_.innerFrames(), H, 1), MTLSizeMake(256, 1, 1));
}

void LinearBranch::readout(id<MTLComputeCommandEncoder> e, const model::DitBlockWeights &w) const {
  BranchParams p = branchParams(lay_);
  p.Spad = x_.rawRows;
  p.row0 = lay_.text;  // compact raw q: inner frames after the text rows
  p.orow0 = 0;
  p.ldo = C;
  p.nrows = lay_.innerRow0();  // output-gate hidden row of inner frame 0
  [e setComputePipelineState:b_.pipeline("vdn_readout")];
  metal::bind(e, x_.rawQ, 0);
  metal::bind(e, x_.state, 1);
  metal::bind(e, x_.stateScale, 2);
  metal::bind(e, x_.outGateHidden, 3);
  metal::bind(e, w.branchNorm, 4);
  metal::bind(e, x_.readout, 5);
  metal::bindBytes(e, p, 6);
  metal::bind(e, w.ogUp, 7);
  metal::bind(e, w.ogBias, 8);
  const int32_t rowsPerGroup = 16 * kReadoutSimdgroups;
  metal::dispatch(e, MTLSizeMake((lay_.tokensPerFrame + rowsPerGroup - 1) / rowsPerGroup, H, lay_.innerFrames()),
                  MTLSizeMake(32 * kReadoutSimdgroups, 1, 1));
}

} // namespace h3::ops
