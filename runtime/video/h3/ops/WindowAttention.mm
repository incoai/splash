#include "video/h3/ops/WindowAttention.hpp"

#include "video/h3/engine/DeviceProfile.hpp"

#include <algorithm>
#include <cmath>

namespace h3::ops {
namespace {

constexpr int32_t H = H3_DIT_HEADS, HD = H3_DIT_HEAD_DIM;

float qkScaleLog2() { return static_cast<float>(1.4426950408889634 / std::sqrt(128.0)); }

// Merge the partials of the key-chunked groups; one launch covers heads [head0, head0 + heads).
void combine(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const AttentionBuffers &a,
             const WindowTable &table, const WindowParams &p, int32_t heads, BufferView out) {
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [e setComputePipelineState:b.pipeline("dit_window_combine")];
  metal::bind(e, a.partials, 0);
  metal::bind(e, a.partialStats, 1);
  metal::bind(e, out, 2);
  metal::bindBytes(e, p, 3);
  metal::bind(e, table.groupBuffer(), 4);
  metal::dispatch(e, MTLSizeMake(static_cast<NSUInteger>(table.chunkedGroups()) * 8, heads, 1),
                  MTLSizeMake(256, 1, 1));
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

template <class Encode>
void launches(const AttentionBuffers &a, const WindowTable &table, int32_t flags, Encode &&encode) {
  const int32_t perLaunch = table.partialSlots() > 0 ? std::min(device::kHeadsPerWindowDispatch, H) : H;
  for (int32_t h0 = 0; h0 < H; h0 += perLaunch) {
    const int32_t heads = std::min(perLaunch, H - h0);
    const WindowParams p{a.rows,
                         a.paddedRows,
                         H,
                         H * HD,
                         a.paddedRows / H3_ATTENTION_KEYS,
                         static_cast<int32_t>(table.items().size()),
                         h0,
                         heads,
                         table.blocks(),
                         table.partialSlots(),
                         flags,
                         qkScaleLog2()};
    encode(p, heads);
  }
}

} // namespace

std::vector<std::string> attentionKernels() {
  return {"dit_attention_key_mean_partial", "dit_attention_key_mean", "dit_attention_quantize",
          "dit_window_attention", "dit_window_combine", "dit_qkfix_window_attention"};
}

void quantize(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const AttentionBuffers &a) {
  const AttentionPrepParams p{a.rows, a.paddedRows, H, device::kKeyBlock, device::kKeyMeanChunks, 1, qkScaleLog2()};
  [e setComputePipelineState:b.pipeline("dit_attention_key_mean_partial")];
  metal::bind(e, a.kHalf, 0);
  metal::bind(e, a.keyMeanPartial, 1);
  metal::bindBytes(e, p, 2);
  metal::dispatch(e, MTLSizeMake(device::kKeyMeanChunks, H, 1), MTLSizeMake(128, 1, 1));
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [e setComputePipelineState:b.pipeline("dit_attention_key_mean")];
  metal::bind(e, a.keyMeanPartial, 0);
  metal::bind(e, a.keyMean, 1);
  metal::bindBytes(e, p, 2);
  metal::dispatch(e, MTLSizeMake(H, 1, 1), MTLSizeMake(128, 1, 1));
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [e setComputePipelineState:b.pipeline("dit_attention_quantize")];
  metal::bind(e, a.qHalf, 0);
  metal::bind(e, a.kHalf, 1);
  metal::bind(e, a.qInt8, 2);
  metal::bind(e, a.kInt8, 3);
  metal::bind(e, a.qScale, 4);
  metal::bind(e, a.kScale, 5);
  metal::bind(e, a.keyMean, 6);
  metal::bindBytes(e, p, 7);
  metal::dispatch(e, MTLSizeMake(a.paddedRows / device::kKeyBlock, H, 2), MTLSizeMake(256, 1, 1));
}

void window(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const AttentionBuffers &a,
            const WindowTable &table, BufferView out) {
  launches(a, table, 0, [&](const WindowParams &p, int32_t heads) {
    [e setComputePipelineState:b.pipeline("dit_window_attention")];
    metal::bind(e, a.qInt8, 0);
    metal::bind(e, a.kInt8, 1);
    metal::bind(e, a.v, 2);
    metal::bind(e, out, 3);
    metal::bind(e, a.qScale, 4);
    metal::bind(e, a.kScale, 5);
    metal::bindBytes(e, p, 6);
    metal::bind(e, table.itemBuffer(), 7);
    metal::bind(e, table.tableBuffer(), 8);
    metal::bind(e, a.partials ? a.partials : out, 9);
    metal::bind(e, a.partialStats ? a.partialStats : out, 10);
    metal::dispatch(e, MTLSizeMake(table.items().size() * heads, 1, 1), MTLSizeMake(128, 1, 1));
    if (table.partialSlots() > 0) combine(e, b, a, table, p, heads, out);
  });
}

void windowQkFix(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const AttentionBuffers &a,
                 BufferView qExact, BufferView kExact, const WindowTable &table, BufferView out) {
  launches(a, table, 0, [&](const WindowParams &p, int32_t heads) {
    [e setComputePipelineState:b.pipeline("dit_qkfix_window_attention")];
    metal::bind(e, a.qHalf, 0);
    metal::bind(e, a.kHalf, 1);
    metal::bind(e, a.v, 2);
    metal::bind(e, out, 3);
    metal::bind(e, qExact, 4);
    metal::bind(e, kExact, 5);
    metal::bindBytes(e, p, 6);
    metal::bind(e, table.itemBuffer(), 7);
    metal::bind(e, table.tableBuffer(), 8);
    metal::bind(e, a.partials ? a.partials : out, 9);
    metal::bind(e, a.partialStats ? a.partialStats : out, 10);
    metal::dispatch(e, MTLSizeMake(table.items().size() * heads, 1, 1), MTLSizeMake(128, 1, 1));
    if (table.partialSlots() > 0) combine(e, b, a, table, p, heads, out);
  });
}

} // namespace h3::ops
