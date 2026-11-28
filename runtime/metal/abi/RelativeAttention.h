#pragma once

// Constants of relative attention (ops::RelativeAttention, kernels/shared/
// relative_attention.metal), shared by host dispatch code and Metal kernels:
// Inkling-2-Mini's attention, whose kernels take SplashAttentionParams
// (metal/abi/PagedAttention.h) as the paged attention's do.
#include "metal/abi/PagedAttention.h"

// Each query head of a row carries SPLASH_RELATIVE_ATTENTION_STATES relative
// states s. A key at backward distance d (the row's position less the key's)
// below SPLASH_RELATIVE_ATTENTION_EXTENT adds s . bias[d] to its score, a
// key past it nothing (the source's d_rel and rel_extent). The layer's bias
// table holds a row of as many states for each distance: [distance][state].
#define SPLASH_RELATIVE_ATTENTION_STATES 16u
#define SPLASH_RELATIVE_ATTENTION_EXTENT 1024u

// The prepare clips each head of the queries and keys to an L2 norm of at
// most SPLASH_RELATIVE_ATTENTION_QK_CLIP (sqrt(512), qk_l2_clip_threshold),
// and of the relative states to SPLASH_RELATIVE_ATTENTION_STATE_CLIP
// (rel_l2_clip_threshold): x min(1, clip / (|x| + epsilon)), the source's
// epsilon. A position-scaled layer's (the full layers') queries and states
// are scaled by 1 + alpha ln(max((position + 1) / floor, 1))
// (log_scaling_alpha, log_scaling_n_floor), which grows their logits past
// the floor's positions.
#define SPLASH_RELATIVE_ATTENTION_QK_CLIP 22.627416997969522f
#define SPLASH_RELATIVE_ATTENTION_STATE_CLIP 8.0f
#define SPLASH_RELATIVE_ATTENTION_CLIP_EPSILON 1e-6f
#define SPLASH_RELATIVE_ATTENTION_LOG_SCALE_ALPHA 0.1f
#define SPLASH_RELATIVE_ATTENTION_LOG_SCALE_FLOOR 8192.0f

// A decode step's single pass (ops::RelativeAttention::addDecode) splits each
// lane's history into at most SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS runs:
// the run that finishes last combines them, a simdgroup lane each.
#define SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS 32u

// A prompt chunk's dispatch whose tiles each read their history in one split
// (ops::RelativeAttention::addAttention) runs them as tiles each of which
// writes its output rows itself (the direct kernels), of the rows the GPU
// family takes (ops::AttentionCalibration::directRows): the most whole rows
// whose fused rows (3 a row) fit the 64 a tile's softmax holds,
// SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS, multiplied as 64, or two of the
// paged attention's tiles, SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS_R16 (the
// kernels named _r16), the same bits. A page's loop costs a threadgroup much
// that does not grow with its rows: at 8 KV heads of 128 a page took 2.17 ms
// a query head over 24 fused rows, 1.91 over 48 and 1.39 over 64 (2,048-row
// chunks after 32K, 20-core M5 Pro).
#define SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS 21u
#define SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS_R16 16u
