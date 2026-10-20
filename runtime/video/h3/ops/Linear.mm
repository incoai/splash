#include "video/h3/ops/Linear.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/LinearABI.h"

#include <algorithm>

namespace h3::ops {
namespace {

constexpr int32_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER, H = H3_DIT_HEADS;
constexpr int32_t kAdaStride = H3_ADA_ROWS * D;

int32_t producerThreads(int32_t K, int32_t groups) { return 16 * (K / 128) / groups; }

template <class Body>
void rowChunks(int32_t rows, Body &&body) {
  for (int32_t r0 = 0; r0 < rows; r0 += device::kMaxRowsPerDispatch)
    body(r0, std::min(device::kMaxRowsPerDispatch, rows - r0));
}

MTLSize gemmGrid(int32_t N, int32_t rows) {
  const device::Tile t = device::kLinearTile;
  return MTLSizeMake((N + t.tn - 1) / t.tn, (rows + t.tm - 1) / t.tm, 1);
}
MTLSize gemmThreads() { return MTLSizeMake(device::kLinearTile.nsg * 32, 1, 1); }

void producer(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const char *kernel, BufferView x,
              int32_t rows, int32_t K, int32_t groups, BufferView normWeight, BufferView ada, AdaRow shift,
              AdaRow scale, BufferView tags, BufferView signs, ProducerOut out) {
  const ProducerParams p{rows, K, kAdaStride, static_cast<int32_t>(shift) * D, static_cast<int32_t>(scale) * D,
                         device::kNormEps};
  [e setComputePipelineState:b.pipeline(kernel)];
  metal::bind(e, x, 0);
  metal::bind(e, out.q, 1);
  metal::bind(e, out.scale, 2);
  metal::bind(e, tags ? tags : x, 3);
  metal::bind(e, normWeight ? normWeight : x, 4);
  metal::bind(e, ada ? ada : x, 5);
  metal::bindBytes(e, p, 6);
  metal::bind(e, signs, 7);
  metal::dispatch(e, MTLSizeMake(rows, 1, 1), MTLSizeMake(producerThreads(K, groups), 1, 1));
}

void innerProducer(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const char *kernel, BufferView x,
                   BufferView gate, int32_t rows, int32_t signOffset, BufferView signs, ProducerOut out) {
  const GatedProducerParams p{rows, C, signOffset, C, H};
  [e setComputePipelineState:b.pipeline(kernel)];
  metal::bind(e, x, 0);
  metal::bind(e, x, 1);
  metal::bind(e, gate, 2);
  metal::bind(e, out.q, 3);
  metal::bind(e, out.scale, 4);
  metal::bind(e, signs, 5);
  metal::bindBytes(e, p, 6);
  metal::dispatch(e, MTLSizeMake(rows, 1, 1), MTLSizeMake(producerThreads(C, H3_PRODUCER_GROUPS_INNER), 1, 1));
}

} // namespace

std::vector<std::string> linearKernels() {
  return {"dit_producer_norm", "dit_producer_plain", "dit_producer_gated", "dit_producer_branch",
          "dit_qkv",           "dit_gated_residual", "dit_fc1_swiglu"};
}

void normQuantize(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, BufferView x, int32_t rows,
                  BufferView normWeight, BufferView ada, AdaRow shift, AdaRow scale, BufferView tags,
                  BufferView signs, ProducerOut out) {
  producer(e, b, "dit_producer_norm", x, rows, D, H3_PRODUCER_GROUPS_HIDDEN, normWeight, ada, shift, scale, tags, signs,
           out);
}

void hiddenQuantize(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, BufferView h, int32_t rows,
                    BufferView signs, ProducerOut out) {
  producer(e, b, "dit_producer_plain", h, rows, F, H3_PRODUCER_GROUPS_FFN, {}, {}, AdaRow::Shift1, AdaRow::Shift1, {},
           signs, out);
}

void gatedQuantize(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, BufferView attention,
                   BufferView softmaxGate, int32_t rows, BufferView signs, ProducerOut out) {
  innerProducer(e, b, "dit_producer_gated", attention, softmaxGate, rows, 0, signs, out);
}

void branchQuantize(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, BufferView readout, int32_t rows,
                    BufferView signs, ProducerOut out) {
  // the gate binding is unused by the branch mode
  innerProducer(e, b, "dit_producer_branch", readout, readout, rows, C, signs, out);
}

void qkv(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, ProducerOut in, int32_t rows,
         BufferView weight, BufferView weightScale, BufferView qNorm, BufferView kNorm, BufferView rope,
         const QkvTargets &out) {
  [e setComputePipelineState:b.pipeline("dit_qkv")];
  rowChunks(rows, [&](int32_t r0, int32_t m) {
    const QkvParams p{m,          H3_DIT_QKV,       D,         H, out.paddedRows, r0, 1 | 2, out.rawRows,
                      device::kQkvSwizzle, out.text, out.innerRow0, out.innerRowEnd, device::kNormEps};
    metal::bind(e, in.q + size_t(r0) * D, 0);
    metal::bind(e, weight, 1);
    metal::bind(e, in.scale + size_t(r0) * 4, 2);
    metal::bind(e, weightScale, 3);
    metal::bind(e, rope + size_t(r0) * H3_DIT_ROPE * 2, 4);
    metal::bind(e, qNorm, 5);
    metal::bind(e, kNorm, 6);
    metal::bind(e, out.q, 7);
    metal::bind(e, out.k, 8);
    metal::bind(e, out.v, 9);
    metal::bind(e, out.rawQ, 10);
    metal::bind(e, out.rawK, 11);
    metal::bindBytes(e, p, 12);
    metal::dispatch(e, gemmGrid(H3_DIT_QKV, m), gemmThreads());
  });
}

void gatedResidual(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, ProducerOut in, int32_t rows,
                   int32_t K, BufferView weight, BufferView weightScale, BufferView x, BufferView tags, BufferView ada,
                   AdaRow gate) {
  [e setComputePipelineState:b.pipeline("dit_gated_residual")];
  rowChunks(rows, [&](int32_t r0, int32_t m) {
    const GatedResidualParams p{m, D, K, D, kAdaStride, static_cast<int32_t>(gate) * D, device::kResidualSwizzle};
    metal::bind(e, in.q + size_t(r0) * K, 0);
    metal::bind(e, weight, 1);
    metal::bind(e, in.scale + size_t(r0) * 4, 2);
    metal::bind(e, weightScale, 3);
    metal::bind(e, x + size_t(r0) * D * 2, 4);
    metal::bind(e, tags + size_t(r0), 5);
    metal::bind(e, ada, 6);
    metal::bindBytes(e, p, 7);
    metal::dispatch(e, gemmGrid(D, m), gemmThreads());
  });
}

void fc1SwiGlu(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, ProducerOut in, int32_t rows,
               BufferView weight, BufferView weightScale, BufferView hidden) {
  [e setComputePipelineState:b.pipeline("dit_fc1_swiglu")];
  rowChunks(rows, [&](int32_t r0, int32_t m) {
    const SwiGluParams p{m, D, F};
    metal::bind(e, in.q + size_t(r0) * D, 0);
    metal::bind(e, weight, 1);
    metal::bind(e, in.scale + size_t(r0) * 4, 2);
    metal::bind(e, weightScale, 3);
    metal::bind(e, hidden + size_t(r0) * F * 2, 4);
    metal::bindBytes(e, p, 5);
    metal::dispatch(e, gemmGrid(F, m), gemmThreads());
  });
}

} // namespace h3::ops
