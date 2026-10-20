#include "video/h3/ops/QkFix.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/QkFixABI.h"

#include <algorithm>

namespace h3::ops {
namespace {
constexpr int32_t D = H3_DIT_HIDDEN, H = H3_DIT_HEADS;
}

std::vector<std::string> qkfixKernels() { return {"dit_qkfix_producer", "dit_qkfix_qkv"}; }

void qkfixNormalize(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, BufferView x, int32_t rows,
                    BufferView normWeight, BufferView ada, BufferView tags, BufferView out) {
  const QkFixProducerParams p{rows, D, H3_ADA_ROWS * D, static_cast<int32_t>(AdaRow::Shift1) * D,
                              static_cast<int32_t>(AdaRow::Scale1) * D, device::kNormEps};
  [e setComputePipelineState:b.pipeline("dit_qkfix_producer")];
  metal::bind(e, x, 0);
  metal::bind(e, out, 1);
  metal::bind(e, tags, 2);
  metal::bind(e, normWeight, 3);
  metal::bind(e, ada, 4);
  metal::bindBytes(e, p, 5);
  metal::dispatch(e, MTLSizeMake(rows, 1, 1), MTLSizeMake(256, 1, 1));
}

void qkfixQkv(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, BufferView in, int32_t rows,
              BufferView weight, BufferView splitDims, BufferView qNorm, BufferView kNorm, BufferView rope,
              const QkFixTargets &out) {
  const int32_t N = H3_DIT_QKV, TM = device::kQkFixTileRows;
  [e setComputePipelineState:b.pipeline("dit_qkfix_qkv")];
  for (int32_t r0 = 0; r0 < rows; r0 += device::kMaxRowsPerDispatch) {
    const int32_t m = std::min(device::kMaxRowsPerDispatch, rows - r0);
    const QkFixQkvParams p{m,         N,  D,           H,
                           out.qkv.paddedRows, r0, 1 | 2, out.qkv.rawRows,
                           out.qkv.text, out.qkv.innerRow0, out.qkv.innerRowEnd, device::kQkvSwizzle,
                           device::kNormEps};
    metal::bind(e, in + size_t(r0) * D * 2, 0);
    metal::bind(e, weight, 1);
    metal::bind(e, rope + size_t(r0) * H3_DIT_ROPE * 2, 2);
    metal::bind(e, qNorm, 3);
    metal::bind(e, kNorm, 4);
    metal::bind(e, out.qkv.q, 5);
    metal::bind(e, out.qkv.k, 6);
    metal::bind(e, out.qkv.rawQ, 7);
    metal::bind(e, out.qkv.rawK, 8);
    metal::bind(e, out.qExact, 9);
    metal::bind(e, out.kExact, 10);
    metal::bind(e, splitDims, 11);
    metal::bindBytes(e, p, 12);
    metal::bind(e, out.qkv.v, 13);
    metal::dispatch(e, MTLSizeMake(N / 128, (m + TM - 1) / TM, 1), MTLSizeMake(128, 1, 1));
  }
}

} // namespace h3::ops
