#include "video/ops/Linear.hpp"

#include "video/metal/DeviceProfile.hpp"
#include "metal/abi/video/LinearABI.h"

#include <algorithm>

namespace splash::video::ops {
namespace {

constexpr int32_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER, H = H3_DIT_HEADS;
constexpr int32_t kAdaStride = H3_ADA_ROWS * D;

int32_t producerThreads(int32_t K, int32_t groups) { return 16 * (K / 128) / groups; }

template <class Body>
void rowChunks(int32_t rows, Body &&body) {
  for (int32_t r0 = 0; r0 < rows; r0 += device::kMaxRowsPerDispatch)
    body(r0, std::min(device::kMaxRowsPerDispatch, rows - r0));
}

metal::DispatchSize gemmGrid(int32_t N, int32_t rows) {
  const device::Tile t = device::kLinearTile;
  return size3((N + t.tn - 1) / t.tn, (rows + t.tm - 1) / t.tm, 1);
}
metal::DispatchSize gemmThreads() { return size3(device::kLinearTile.nsg * 32, 1, 1); }

// The up-projection's parameters of a row chunk of m rows; T's view of the chunk at r0.
LoraUpParams loraUp(const LoraBinding &lora, int32_t m, int32_t segN) {
  return {lora.ldt, (m + 63) / 64 * 64, segN, lora.segments};
}
BufferView loraRows(const LoraBinding &lora, int32_t r0) { return lora.t + uint64_t(r0) * lora.ldt * 2; }

void loraDownRows(Encoder &e, const char *kernel, BufferView x, size_t elementBytes, BufferView rowScale,
                  int32_t rows, int32_t K, const LoraBinding &lora) {
  const int32_t R = lora.segments * H3_LORA_RANK;
  e.pipeline(kernel);
  rowChunks(rows, [&](int32_t r0, int32_t m) {
    const LoraDownParams p{m, K, R, lora.ldt, lora.scale, 0, 0, 0};
    e.bind(x + uint64_t(r0) * K * elementBytes, 0);
    e.bind(lora.a, 1);
    e.bind(rowScale ? rowScale + uint64_t(r0) * 4 : lora.a, 2);
    e.bind(loraRows(lora, r0), 3);
    e.bytes(p, 4);
    e.dispatch(size3(R / H3_LORA_RANK, (m + device::kLinearTile.tm - 1) / device::kLinearTile.tm, 1), gemmThreads());
  });
}

void producer(Encoder &e, const char *kernel, BufferView x, int32_t rows, int32_t K, int32_t groups,
              BufferView normWeight, BufferView ada, AdaRow shift, AdaRow scale, BufferView tags, BufferView signs,
              ProducerOut out) {
  const ProducerParams p{rows, K, kAdaStride, static_cast<int32_t>(shift) * D, static_cast<int32_t>(scale) * D,
                         device::kNormEps};
  e.pipeline(kernel);
  e.bind(x, 0);
  e.bind(out.q, 1);
  e.bind(out.scale, 2);
  e.bind(tags ? tags : x, 3);
  e.bind(normWeight ? normWeight : x, 4);
  e.bind(ada ? ada : x, 5);
  e.bytes(p, 6);
  e.bind(signs, 7);
  e.dispatch(size3(rows, 1, 1), size3(producerThreads(K, groups), 1, 1));
}

void innerProducer(Encoder &e, const char *kernel, BufferView x, BufferView gate, int32_t rows, int32_t signOffset,
                   BufferView signs, ProducerOut out) {
  const GatedProducerParams p{rows, C, signOffset, C, H};
  e.pipeline(kernel);
  e.bind(x, 0);
  e.bind(x, 1);
  e.bind(gate, 2);
  e.bind(out.q, 3);
  e.bind(out.scale, 4);
  e.bind(signs, 5);
  e.bytes(p, 6);
  e.dispatch(size3(rows, 1, 1), size3(producerThreads(C, H3_PRODUCER_GROUPS_INNER), 1, 1));
}

} // namespace

void loraDown(Encoder &e, ProducerOut in, int32_t rows, int32_t K, const LoraBinding &lora) {
  loraDownRows(e, "dit_lora_down", in.q, 1, in.scale, rows, K, lora);
}

void loraDownHalf(Encoder &e, BufferView x, int32_t rows, int32_t K, const LoraBinding &lora) {
  loraDownRows(e, "dit_lora_down_f16", x, 2, {}, rows, K, lora);
}

void normQuantize(Encoder &e, BufferView x, int32_t rows, BufferView normWeight, BufferView ada, AdaRow shift,
                  AdaRow scale, BufferView tags, BufferView signs, ProducerOut out) {
  producer(e, "dit_producer_norm", x, rows, D, H3_PRODUCER_GROUPS_HIDDEN, normWeight, ada, shift, scale, tags, signs,
           out);
}

void hiddenQuantize(Encoder &e, BufferView h, int32_t rows, BufferView signs, ProducerOut out) {
  producer(e, "dit_producer_plain", h, rows, F, H3_PRODUCER_GROUPS_FFN, {}, {}, AdaRow::Shift1, AdaRow::Shift1, {},
           signs, out);
}

void gatedQuantize(Encoder &e, BufferView attention, BufferView softmaxGate, int32_t rows, BufferView signs,
                   ProducerOut out) {
  innerProducer(e, "dit_producer_gated", attention, softmaxGate, rows, 0, signs, out);
}

void branchQuantize(Encoder &e, BufferView readout, int32_t rows, BufferView signs, ProducerOut out) {
  // the gate binding is unused by the branch mode
  innerProducer(e, "dit_producer_branch", readout, readout, rows, C, signs, out);
}

void qkv(Encoder &e, ProducerOut in, int32_t rows, BufferView weight, BufferView weightScale, BufferView qNorm,
         BufferView kNorm, BufferView rope, const QkvTargets &out, const LoraBinding &lora) {
  if (lora) loraDown(e, in, rows, D, lora);
  e.pipeline(lora ? "dit_qkv_lora_post" : "dit_qkv");
  rowChunks(rows, [&](int32_t r0, int32_t m) {
    const QkvParams p{m,          H3_DIT_QKV,       D,         H, out.paddedRows, r0, 1 | 2, out.rawRows,
                      device::kQkvSwizzle, out.text, out.innerRow0, out.innerRowEnd, device::kNormEps, 0, 0};
    e.bind(in.q + size_t(r0) * D, 0);
    e.bind(weight, 1);
    e.bind(in.scale + size_t(r0) * 4, 2);
    e.bind(weightScale, 3);
    e.bind(rope + size_t(r0) * H3_DIT_ROPE * 2, 4);
    e.bind(qNorm, 5);
    e.bind(kNorm, 6);
    e.bind(out.q, 7);
    e.bind(out.k, 8);
    e.bind(out.v, 9);
    e.bind(out.rawQ, 10);
    e.bind(out.rawK, 11);
    e.bytes(p, 12);
    e.bind(out.v, 13);  // FastH3's gate target, unused (N covers q / k / v only)
    if (lora) {
      e.bind(loraRows(lora, r0), 14);
      e.bind(lora.b, 15);
      e.bytes(loraUp(lora, m, C), 16);
    }
    e.dispatch(gemmGrid(H3_DIT_QKV, m), gemmThreads());
  });
}

void gatedResidual(Encoder &e, ProducerOut in, int32_t rows, int32_t K, BufferView weight, BufferView weightScale,
                   BufferView x, BufferView tags, BufferView ada, AdaRow gate, const LoraBinding &lora) {
  if (lora) loraDown(e, in, rows, K, lora);
  e.pipeline(lora ? "dit_gated_residual_lora_post" : "dit_gated_residual");
  rowChunks(rows, [&](int32_t r0, int32_t m) {
    const GatedResidualParams p{m, D, K, D, kAdaStride, static_cast<int32_t>(gate) * D, device::kResidualSwizzle};
    e.bind(in.q + size_t(r0) * K, 0);
    e.bind(weight, 1);
    e.bind(in.scale + size_t(r0) * 4, 2);
    e.bind(weightScale, 3);
    e.bind(x + size_t(r0) * D * 2, 4);
    e.bind(tags + size_t(r0), 5);
    e.bind(ada, 6);
    e.bytes(p, 7);
    if (lora) {
      e.bind(loraRows(lora, r0), 8);
      e.bind(lora.b, 9);
      e.bytes(loraUp(lora, m, D), 10);
    }
    e.dispatch(gemmGrid(D, m), gemmThreads());
  });
}

void fc1SwiGlu(Encoder &e, ProducerOut in, int32_t rows, BufferView weight, BufferView weightScale, BufferView hidden,
               const LoraBinding &lora) {
  if (lora) loraDown(e, in, rows, D, lora);
  e.pipeline(lora ? "dit_fc1_swiglu_lora_post" : "dit_fc1_swiglu");
  rowChunks(rows, [&](int32_t r0, int32_t m) {
    const SwiGluParams p{m, D, F};
    e.bind(in.q + size_t(r0) * D, 0);
    e.bind(weight, 1);
    e.bind(in.scale + size_t(r0) * 4, 2);
    e.bind(weightScale, 3);
    e.bind(hidden + size_t(r0) * F * 2, 4);
    e.bytes(p, 5);
    if (lora) {
      e.bind(loraRows(lora, r0), 6);
      e.bind(lora.b, 7);
      e.bytes(loraUp(lora, m, 2 * F), 8);
    }
    e.dispatch(gemmGrid(F, m), gemmThreads());
  });
}

} // namespace splash::video::ops
