#include "video/h3/ops/LinearBranch.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/VdnABI.h"

namespace h3::ops {
namespace {
constexpr int32_t D = H3_DIT_HIDDEN, C = H3_DIT_INNER, H = H3_DIT_HEADS, G = H3_DIT_GATES;
constexpr int32_t kReadoutSimdgroups = 2;
}  // namespace

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

void LinearBranch::gates(Encoder &e, ProducerOut in, BufferView qkvg, BufferView qkvgScale, BufferView sgBias) const {
  const int32_t M = lay_.rows();
  const BranchGateParams p{M, G, D};
  e.pipeline("vdn_gates");
  e.bind(in.q, 0);
  e.bind(qkvg + size_t(H3_DIT_QKV) * D, 1);
  e.bind(in.scale, 2);
  e.bind(qkvgScale + size_t(H3_DIT_QKV) * 4, 3);
  e.bind(x_.beta, 4);
  e.bind(x_.softmaxGate, 5);
  e.bind(x_.outGateHidden, 6);
  e.bind(sgBias, 7);
  e.bytes(p, 8);
  e.dispatch(size3(G / 128, (M + 63) / 64, 1), size3(128, 1, 1));
}

void LinearBranch::frameMean(Encoder &e, ProducerOut in) const {
  const BranchFrameParams p{lay_.tokensPerFrame, D, H, lay_.innerFrames(), lay_.innerRow0()};
  e.pipeline("vdn_frame_mean");
  e.bind(in.q, 0);
  e.bind(in.scale, 1);
  e.bind(x_.frameMean, 2);
  e.bytes(p, 3);
  e.dispatch(size3((D / 4 + 255) / 256, lay_.innerFrames(), 1), size3(256, 1, 1));
}

void LinearBranch::alpha(Encoder &e, const model::DitBlockWeights &w) const {
  const BranchFrameParams p{lay_.tokensPerFrame, D, H, lay_.innerFrames(), lay_.innerRow0()};
  e.pipeline("vdn_alpha");
  e.bind(x_.frameMean, 0);
  e.bind(w.alphaDown, 1);
  e.bind(w.alphaUp, 2);
  e.bind(w.dtBias, 3);
  e.bind(w.aLog, 4);
  e.bind(x_.alpha, 5);
  e.bytes(p, 6);
  e.dispatch(size3(lay_.innerFrames(), 1, 1), size3(1024, 1, 1));
}

// text rows of raw k (L2) and of V: the state start of both scans
void LinearBranch::textFeatures(Encoder &e) const {
  for (int kv = 0; kv < 2; ++kv) {
    BranchParams p = branchParams(lay_);
    p.row0 = 0;
    p.l2 = kv == 0;
    p.nrows = lay_.text;
    p.Spad = kv == 0 ? x_.rawRows : x_.paddedRows;
    e.pipeline("vdn_text_features");
    e.bind(kv == 0 ? x_.rawK : x_.v, 0);
    e.bind(kv == 0 ? x_.textKeys : x_.textValues, 1);
    e.bind(x_.beta, 2);
    e.bytes(p, 3);
    e.dispatch(size3((lay_.text + 7) / 8, H, 1), size3(256, 1, 1));
  }
}

void LinearBranch::features(Encoder &e, bool key, BufferView spatial, BufferView temporal) const {
  BranchParams p = branchParams(lay_);
  p.l2 = key;
  p.Spad = key ? x_.rawRows : x_.paddedRows;
  p.row0 = key ? lay_.text : lay_.innerRow0();  // raw k is compact: inner frames start after the text rows
  e.pipeline("vdn_conv_features");
  e.bind(key ? x_.rawK : x_.v, 0);
  e.bind(key ? x_.keyFeatures : x_.valueFeatures, 1);
  e.bind(spatial, 2);
  e.bind(temporal, 3);
  e.bind(x_.beta + size_t(lay_.innerRow0()) * H * 4, 4);
  e.bytes(p, 5);
  const int32_t tiles = ((lay_.gridH + 5) / 6) * ((lay_.gridW + 5) / 6);  // 6 x 6 token tiles
  e.dispatch(size3(tiles, H, 1), size3(32 * 3 * 6, 1, 1));
}

void LinearBranch::keyFeatures(Encoder &e, const model::DitBlockWeights &w) const {
  features(e, true, w.kSpatial, w.kTemporal);
}

void LinearBranch::valueFeatures(Encoder &e, const model::DitBlockWeights &w) const {
  features(e, false, w.vSpatial, w.vTemporal);
}

void LinearBranch::stats(Encoder &e) const {
  const BranchParams p = branchParams(lay_);
  e.pipeline("vdn_stats");
  e.bind(x_.keyFeatures, 0);
  e.bind(x_.valueFeatures, 1);
  e.bind(x_.statsA, 2);
  e.bind(x_.statsB, 3);
  e.bytes(p, 4);
  e.dispatch(size3(4, H, lay_.innerFrames()), size3(128, 1, 1));
}

void LinearBranch::textStats(Encoder &e) const {
  BranchParams p = branchParams(lay_);
  p.Fp = 1;
  p.S = lay_.text;
  e.pipeline("vdn_stats");
  e.bind(x_.textKeys, 0);
  e.bind(x_.textValues, 1);
  e.bind(x_.textA, 2);
  e.bind(x_.textB, 3);
  e.bytes(p, 4);
  e.dispatch(size3(4, H, 1), size3(128, 1, 1));
}

void LinearBranch::inverse(Encoder &e) const {
  e.pipeline("vdn_inverse");
  e.bind(x_.statsA, 0);
  e.bind(x_.inverse, 1);
  e.bind(x_.pivots, 2);
  e.dispatch(size3(lay_.innerFrames() * H, 1, 1), size3(1024, 1, 1));
}

void LinearBranch::textInverse(Encoder &e) const {
  e.pipeline("vdn_inverse");
  e.bind(x_.textA, 0);
  e.bind(x_.textInverse, 1);
  e.bind(x_.textPivots, 2);
  e.dispatch(size3(H, 1, 1), size3(1024, 1, 1));
}

void LinearBranch::scan(Encoder &e) const {
  const BranchParams p = branchParams(lay_);
  e.pipeline("vdn_scan");
  e.bind(x_.inverse, 0);
  e.bind(x_.statsB, 1);
  e.bind(x_.alpha, 2);
  e.bind(x_.scanPre, 3);
  e.bind(x_.scanSuf, 4);
  e.bind(x_.textInverse, 5);
  e.bind(x_.textB, 6);
  e.bind(x_.scanStart, 7);
  e.bind(x_.scanScratch, 8);
  e.bytes(p, 9);
  e.dispatch(size3(H, 2, 1), size3(1024, 1, 1));
}

void LinearBranch::gather(Encoder &e) const {
  const BranchParams p = branchParams(lay_);
  e.pipeline("vdn_gather");
  e.bind(x_.scanPre, 0);
  e.bind(x_.scanSuf, 1);
  e.bind(x_.alpha, 2);
  e.bind(x_.scanStart, 3);
  e.bind(x_.bounds, 4);
  e.bind(x_.state, 5);
  e.bind(x_.stateScale, 6);
  e.bytes(p, 7);
  e.dispatch(size3(lay_.innerFrames(), H, 1), size3(256, 1, 1));
}

void LinearBranch::readout(Encoder &e, const model::DitBlockWeights &w) const {
  BranchParams p = branchParams(lay_);
  p.Spad = x_.rawRows;
  p.row0 = lay_.text;  // compact raw q: inner frames after the text rows
  p.orow0 = 0;
  p.ldo = C;
  p.nrows = lay_.innerRow0();  // output-gate hidden row of inner frame 0
  e.pipeline("vdn_readout");
  e.bind(x_.rawQ, 0);
  e.bind(x_.state, 1);
  e.bind(x_.stateScale, 2);
  e.bind(x_.outGateHidden, 3);
  e.bind(w.branchNorm, 4);
  e.bind(x_.readout, 5);
  e.bytes(p, 6);
  e.bind(w.ogUp, 7);
  e.bind(w.ogBias, 8);
  const int32_t rowsPerGroup = 16 * kReadoutSimdgroups;
  e.dispatch(size3((lay_.tokensPerFrame + rowsPerGroup - 1) / rowsPerGroup, H, lay_.innerFrames()),
             size3(32 * kReadoutSimdgroups, 1, 1));
}

} // namespace h3::ops
