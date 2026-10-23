// Start-up check of the cooperative-tensor slot formula (common/window.h) the attention kernels rely on, against the
// MPP accessors: QK destination 16 x 32 (int8 and fp16 Q / K, head dim 128; fp16 head dim 64), P left input 16 x 32
// and PV destination 16 x D. out[lane * 3 + config] = mismatches (+1000 for a capacity mismatch); all must be 0.
#include "metal/kernels/video/common/window.h"

using namespace h3;

template <class QT, class ACC, int D>
static inline int slot_mismatches(device QT *base, ushort lane) {
  constexpr int BK = 32;
  auto Q = mk<QT>(base, D, 64);
  auto mQ = Q.template slice<D, 16>(0, 0);
  auto mK = Q.template slice<D, BK>(0, 0);
  constexpr auto dqk = matmul2d_descriptor(16, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(16, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto sT = opqk.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK), ACC>();
  auto pL = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V = mk<half>((device half *)base, D, 64);
  auto V0 = V.template slice<D, BK>(0, 0);
  auto oT = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL)>, decltype(V0), float>();
  const int r0 = lane_r0(lane), cb = lane_cb(lane);
  int bad = 0;
  if (sT.get_capacity() != 16 * BK / 32 || pL.get_capacity() != 16 * BK / 32 || oT.get_capacity() != 16 * D / 32)
    bad += 1000;
  for (ushort i = 0; i < sT.get_capacity(); ++i) {
    int row = r0 + 8 * ((i >> 2) & 1), col = 16 * (i >> 3) + cb + (i & 3);
    auto a = sT.get_multidimensional_index(i);
    auto b = pL.get_multidimensional_index(i);
    bad += (a[0] != col || a[1] != row || !sT.is_valid_element(i));
    bad += (b[0] != col || b[1] != row || !pL.is_valid_element(i));
  }
  for (ushort i = 0; i < oT.get_capacity(); ++i) {
    int row = r0 + 8 * ((i >> 2) & 1), col = 16 * (i >> 3) + cb + (i & 3);
    auto a = oT.get_multidimensional_index(i);
    bad += (a[0] != col || a[1] != row || !oT.is_valid_element(i));
  }
  return bad;
}

// One simdgroup; buf: 64 x 128 fp16 scratch (contents unused).
kernel void layout_check(device uchar *buf [[buffer(0)]], device int *out [[buffer(1)]],
                         ushort lane [[thread_index_in_simdgroup]]) {
  out[lane * 3 + 0] = slot_mismatches<int8_t, int, 128>((device int8_t *)buf, lane);
  out[lane * 3 + 1] = slot_mismatches<half, float, 128>((device half *)buf, lane);
  out[lane * 3 + 2] = slot_mismatches<half, float, 64>((device half *)buf, lane);
}
