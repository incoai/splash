#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The LiLiCorr selector of Bolt's draft (llama.cpp build_lilicorr_scores):
// each proposal position's eight best draft candidates are nodes of a small
// transformer, 56 to a lane, whose factor heads score every edge from a
// position's candidate to the next one's, and from the lane's anchor to the
// first position's.
#define SPLASH_LILICORR_CANDIDATES 8u
#define SPLASH_LILICORR_NODES (SPLASH_DRAFT_PROPOSAL_TOKENS * SPLASH_LILICORR_CANDIDATES)
#define SPLASH_LILICORR_HIDDEN 1024u
#define SPLASH_LILICORR_HEADS 8u
#define SPLASH_LILICORR_HEAD_DIMENSION (SPLASH_LILICORR_HIDDEN / SPLASH_LILICORR_HEADS)
#define SPLASH_LILICORR_MLP 2048u
#define SPLASH_LILICORR_LAYERS 2u
// A candidate's features: log p, p, log p - log p of the best, rank / 7 and
// whether it is the best.
#define SPLASH_LILICORR_FEATURES 5u
#define SPLASH_LILICORR_LOGIT_SCALE 8.0f
#define SPLASH_LILICORR_VECTOR_EPSILON 1e-4f
#define SPLASH_LILICORR_FEATURE_EPSILON 1e-5f
