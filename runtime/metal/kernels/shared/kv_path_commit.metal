#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_path_commit.h"
#include "metal/kernels/common/lane_bindings.h"

// The path commit (kernels/common/kv_path_commit.h) of the pages of each
// layout the paged attention stores.
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit, 4, 256, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv2_g8, 2, 256, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit, 4, 256, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv2_g8, 2, 256, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv4_d128, 4, 128, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv4_d128, 4, 128, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv2_vt, 2, 256, int8_t, true)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv2_vt, 2, 256, bfloat, true)
#undef PAGED_ATTENTION_PATH_COMMIT
