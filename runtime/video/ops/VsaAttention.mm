#include "video/ops/VsaAttention.hpp"

#include "metal/abi/video/LinearABI.h"
#include "metal/abi/video/VsaABI.h"
#include "video/metal/DeviceProfile.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::video::ops {
namespace {

constexpr int32_t D = H3_DIT_HIDDEN, C = H3_DIT_INNER, H = H3_DIT_HEADS;

VsaParams params(const VsaTable &t, int32_t rows, int32_t paddedRows, int32_t flags) {
  VsaParams p{};
  p.M = rows;
  p.Spad = paddedRows;
  p.H = H;
  p.T = t.tiles();
  p.Tp = t.paddedTiles();
  p.V = t.videoTiles();
  p.Nv = t.videoRows();
  p.topk = t.topk();
  p.nkb = paddedRows / H3_ATTENTION_KEYS;
  p.nitems = static_cast<int32_t>(t.items().size());
  p.head0 = 0;
  p.hb = H;
  p.flags = flags;
  p.score_scale = static_cast<float>(1.0 / std::sqrt(128.0));
  p.qk_scale = static_cast<float>(1.4426950408889634 / std::sqrt(128.0));
  return p;
}

} // namespace

void qkvGate(Encoder &e, ProducerOut in, int32_t rows, BufferView weight, BufferView weightScale, BufferView qNorm,
             BufferView kNorm, BufferView rope, const QkvTargets &out, BufferView gate, QkvRows parts,
             const LoraBinding &lora) {
  const bool gateOnly = parts == QkvRows::GateOnly;
  const int32_t n = gateOnly ? C : (parts == QkvRows::QkvOnly ? H3_DIT_QKV : H3_DIT_QKV + C);
  const uint64_t w0 = gateOnly ? uint64_t(H3_DIT_QKV) : 0;  // first weight row
  const device::Tile tile = device::kLinearTile;
  if (lora && gateOnly) throw std::logic_error("qkvGate: a LoRA of the gate rows alone");
  if (lora) loraDown(e, in, rows, D, lora);
  e.pipeline(lora ? "dit_qkv_lora_post" : "dit_qkv");
  for (int32_t r0 = 0; r0 < rows; r0 += device::kMaxRowsPerDispatch) {
    const int32_t m = std::min(device::kMaxRowsPerDispatch, rows - r0);
    const QkvParams p{m, n, D, H, out.paddedRows, r0, 0, 0, device::kQkvSwizzle, 0, 0, 0, device::kNormEps,
                      gateOnly ? 3 : 0, C};
    e.bind(in.q + uint64_t(r0) * D, 0);
    e.bind(weight + w0 * D, 1);
    e.bind(in.scale + uint64_t(r0) * 4, 2);
    e.bind(weightScale + w0 * 4, 3);
    e.bind(rope + uint64_t(r0) * H3_DIT_ROPE * 2, 4);
    e.bind(qNorm, 5);
    e.bind(kNorm, 6);
    e.bind(out.q, 7);
    e.bind(out.k, 8);
    e.bind(out.v, 9);
    e.bind(out.v, 10);  // no raw q / k (flags 0)
    e.bind(out.v, 11);
    e.bytes(p, 12);
    e.bind(gate ? gate : out.v, 13);  // QkvOnly: no gate rows, the binding unused
    if (lora) {
      e.bind(lora.t + uint64_t(r0) * lora.ldt * 2, 14);
      e.bind(lora.b, 15);
      e.bytes(LoraUpParams{lora.ldt, (m + 63) / 64 * 64, C, lora.segments}, 16);
    }
    e.dispatch(size3((n + tile.tn - 1) / tile.tn, (m + tile.tm - 1) / tile.tm, 1), size3(tile.nsg * 32, 1, 1));
  }
}

void vsaPermute(Encoder &e, BufferView src, BufferView dst, const metal::MetalBuffer &index, int32_t rows) {
  const VsaPermuteParams p{rows, D};
  e.pipeline("dit_vsa_permute");
  e.bind(src, 0);
  e.bind(dst, 1);
  e.bind(index, 2);
  e.bytes(p, 3);
  e.dispatch(size3(rows, 1, 1), size3(224, 1, 1));
}

void vsaPool(Encoder &e, const VsaTable &t, const AttentionBuffers &a, BufferView qExact, BufferView kExact,
             BufferView splitDims, const VsaBuffers &v) {
  const bool qkfix = static_cast<bool>(splitDims);
  const VsaParams p = params(t, a.rows, a.paddedRows, qkfix ? 1 : 0);
  e.pipeline("dit_vsa_pool");
  e.bind(a.qHalf, 0);
  e.bind(a.kHalf, 1);
  e.bind(a.v, 2);
  e.bind(qkfix ? qExact : a.qHalf, 3);
  e.bind(qkfix ? kExact : a.kHalf, 4);
  e.bind(qkfix ? splitDims : a.qHalf, 5);
  e.bind(t.tileBuffer(), 6);
  e.bind(v.pooledQ, 7);
  e.bind(v.pooledK, 8);
  e.bind(v.pooledV, 9);
  e.bytes(p, 10);
  e.dispatch(size3(t.tiles(), H, 1), size3(128, 1, 1));
}

void vsaRoute(Encoder &e, const VsaTable &t, const AttentionBuffers &a, const VsaBuffers &v) {
  const VsaParams p = params(t, a.rows, a.paddedRows, 0);
  const uint64_t blocks = static_cast<uint64_t>(p.Tp / 32);
  e.pipeline("dit_vsa_scores");
  e.bind(v.pooledQ, 0);
  e.bind(v.pooledK, 1);
  e.bind(v.scores, 2);
  e.bytes(p, 3);
  e.dispatch(size3(blocks, blocks, H), size3(128, 1, 1));
  e.pipeline("dit_vsa_route");
  e.bind(v.scores, 0);
  e.bind(v.selected, 1);
  e.bytes(p, 2);
  e.dispatch(size3(t.tiles(), H, 1), size3(H3_VSA_ROUTE_THREADS, 1, 1));
  e.pipeline("dit_vsa_compress");
  e.bind(v.scores, 0);
  e.bind(v.pooledV, 1);
  e.bind(v.outC, 2);
  e.bytes(p, 3);
  e.dispatch(size3(blocks, 4, H), size3(128, 1, 1));
}

void vsaAttention(Encoder &e, const VsaTable &t, const AttentionBuffers &a, BufferView qExact, BufferView kExact,
                  const VsaBuffers &v, BufferView out) {
  const bool qkfix = static_cast<bool>(qExact), dense = t.spec().dense;
  VsaParams p = params(t, a.rows, a.paddedRows, (qkfix ? 1 : 0) | (dense ? 2 : 0));
  if (dense) p.nitems = t.denseItems();
  if (qkfix) {
    e.pipeline(dense ? "dit_vsa_attention_qkfix_dense" : "dit_vsa_attention_qkfix");
    e.bind(a.qHalf, 0);
    e.bind(a.kHalf, 1);
    e.bind(a.v, 2);
    e.bind(out, 3);
    e.bind(qExact, 4);
    e.bind(kExact, 5);
  } else {
    e.pipeline(dense ? "dit_vsa_attention_dense" : "dit_vsa_attention");
    e.bind(a.qInt8, 0);
    e.bind(a.kInt8, 1);
    e.bind(a.v, 2);
    e.bind(out, 3);
    e.bind(a.qScale, 4);
    e.bind(a.kScale, 5);
  }
  e.bytes(p, 6);
  e.bind(dense ? t.denseItemBuffer() : t.itemBuffer(), 7);
  e.bind(dense ? t.denseTileBuffer() : t.tileBuffer(), 8);
  e.bind(dense ? t.denseTileBuffer() : v.selected, 9);  // dense: no selection read
  e.dispatch(size3(uint64_t(p.nitems) * H, 1, 1), size3(128, 1, 1));
}

void vsaQuantize(Encoder &e, const VsaTable &t, BufferView attention, BufferView gate, const VsaBuffers &v,
                 int32_t rows, int32_t paddedRows, BufferView signs, ProducerOut out) {
  const bool dense = t.spec().dense;
  const VsaParams p = params(t, rows, paddedRows, dense ? 2 : 0);
  e.pipeline("dit_producer_vsa");
  e.bind(attention, 0);
  e.bind(dense ? attention : gate, 1);  // dense: O alone, gate and out_c unread
  e.bind(dense ? attention : v.outC, 2);
  e.bind(t.rowTileBuffer(), 3);
  e.bind(out.q, 4);
  e.bind(out.scale, 5);
  e.bind(signs, 6);
  e.bytes(p, 7);
  // dit_producer_gated's geometry: 16 K / 128 / G threads per row
  e.dispatch(size3(rows, 1, 1), size3(16 * (C / 128) / H3_PRODUCER_GROUPS_INNER, 1, 1));
}

} // namespace splash::video::ops
