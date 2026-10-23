#include "video/ops/QkFix.hpp"

#include "video/metal/DeviceProfile.hpp"
#include "metal/abi/video/QkFixABI.h"

#include <algorithm>

namespace splash::video::ops {
namespace {
constexpr int32_t D = H3_DIT_HIDDEN, H = H3_DIT_HEADS;
}

void qkfixNormalize(Encoder &e, BufferView x, int32_t rows, BufferView normWeight, BufferView ada, BufferView tags,
                    BufferView out) {
  const QkFixProducerParams p{rows, D, H3_ADA_ROWS * D, static_cast<int32_t>(AdaRow::Shift1) * D,
                              static_cast<int32_t>(AdaRow::Scale1) * D, device::kNormEps};
  e.pipeline("dit_qkfix_producer");
  e.bind(x, 0);
  e.bind(out, 1);
  e.bind(tags, 2);
  e.bind(normWeight, 3);
  e.bind(ada, 4);
  e.bytes(p, 5);
  e.dispatch(size3(rows, 1, 1), size3(256, 1, 1));
}

void qkfixQkv(Encoder &e, BufferView in, int32_t rows, BufferView weight, BufferView splitDims, BufferView qNorm,
              BufferView kNorm, BufferView rope, const QkFixTargets &out) {
  const int32_t N = H3_DIT_QKV, TM = device::kQkFixTileRows;
  e.pipeline("dit_qkfix_qkv");
  for (int32_t r0 = 0; r0 < rows; r0 += device::kMaxRowsPerDispatch) {
    const int32_t m = std::min(device::kMaxRowsPerDispatch, rows - r0);
    const QkFixQkvParams p{m,         N,  D,           H,
                           out.qkv.paddedRows, r0, 1 | 2, out.qkv.rawRows,
                           out.qkv.text, out.qkv.innerRow0, out.qkv.innerRowEnd, device::kQkvSwizzle,
                           device::kNormEps};
    e.bind(in + size_t(r0) * D * 2, 0);
    e.bind(weight, 1);
    e.bind(rope + size_t(r0) * H3_DIT_ROPE * 2, 2);
    e.bind(qNorm, 3);
    e.bind(kNorm, 4);
    e.bind(out.qkv.q, 5);
    e.bind(out.qkv.k, 6);
    e.bind(out.qkv.rawQ, 7);
    e.bind(out.qkv.rawK, 8);
    e.bind(out.qExact, 9);
    e.bind(out.kExact, 10);
    e.bind(splitDims, 11);
    e.bytes(p, 12);
    e.bind(out.qkv.v, 13);
    e.dispatch(size3(N / 128, (m + TM - 1) / TM, 1), size3(128, 1, 1));
  }
}

} // namespace splash::video::ops
