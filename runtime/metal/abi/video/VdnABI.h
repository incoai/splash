#pragma once

// VDN linear branch (OpenVDN vdn-minimax-h3): gate projections, frame means and decays, key / value features,
// per-frame statistics, (I + A)^-1, forward / reverse scans, window gather and the gated readout.
#include "metal/abi/video/Layout.h"

// Feature / statistics / scan / gather / readout kernels.
struct BranchParams {
  int32_t H, S, gh, gw;          // heads, tokens per frame, token grid
  int32_t Fp, Spad, row0, l2;    // inner frames, rows per head of the raw buffer, first raw row, L2-normalise flag
  int32_t ldo, orow0, nrows;     // readout row stride / first output row; plain features: rows; readout: og row 0
  float eps;
};

// vdn_frame_mean / vdn_alpha
struct BranchFrameParams {
  int32_t S, K, H, Fp;
  int32_t row0;
};

// vdn_gates: W8A8 [M, 256] on the QKV input
struct BranchGateParams {
  int32_t M, N, K;
};
