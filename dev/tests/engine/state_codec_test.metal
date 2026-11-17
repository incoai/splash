#include "metal/kernels/common/state_codec.h"

// The codes the decode fold stores of a state (state_code_quad), for the
// state codec's test: thread `id` holds values [4 id, 4 id + 4), as a lane of
// the decode scan holds four of a row's key columns, eight a block, and
// stores their codes at the same offsets and, as its block's first lane, the
// block's scale at `scales_at` past the codes, the codec's layout; State is
// the running state's cell. Compiled with the production flags: the decode
// kernels build with fast math, under which max() may drop a NaN.
template <class State>
inline void splash_test_state_code_quad(device const float *values, device char *codes, constant uint &scales_at,
                                        uint id, uint lane) {
  float quad[4];
  for (uint i = 0; i < 4; ++i)
    quad[i] = values[4 * id + i];
  state_code_quad<State>(quad, codes + 4 * id, reinterpret_cast<device bfloat *>(codes + scales_at) + id / 8, lane);
}

#define TEST_STATE_CODE_QUAD(Name, State)                                                                      \
  kernel void Name(device const float *values [[buffer(0)]], device char *codes [[buffer(1)]],               \
                   constant uint &scales_at [[buffer(2)]], uint id [[thread_position_in_grid]],             \
                   uint lane [[thread_index_in_simdgroup]]) {                                                \
    splash_test_state_code_quad<State>(values, codes, scales_at, id, lane);                                  \
  }
TEST_STATE_CODE_QUAD(test_state_code_quad_f32, float)
TEST_STATE_CODE_QUAD(test_state_code_quad_bf16, bfloat)
#undef TEST_STATE_CODE_QUAD
