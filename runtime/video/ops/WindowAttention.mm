#include "video/ops/WindowAttention.hpp"

#include "video/metal/DeviceProfile.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace splash::video::ops {
namespace {

constexpr int32_t H = H3_DIT_HEADS, HD = H3_DIT_HEAD_DIM;

float qkScaleLog2() { return static_cast<float>(1.4426950408889634 / std::sqrt(128.0)); }

// Merge the partials of the key-chunked groups; one launch covers heads [head0, head0 + heads).
void combine(Encoder &e, const AttentionBuffers &a, const WindowTable &table, const WindowParams &p, int32_t heads,
             BufferView out) {
  e.pipeline("dit_window_combine");
  e.bind(a.partials, 0);
  e.bind(a.partialStats, 1);
  e.bind(out, 2);
  e.bytes(p, 3);
  e.bind(table.groupBuffer(), 4);
  e.dispatch(size3(uint64_t(table.chunkedGroups()) * 8, heads), size3(256));
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

void checkRegisterLayout(metal::MetalBackend &backend) {
  const metal::MetalBuffer scratch = allocate(backend, 64 * 128 * 2, "layout check");
  const metal::MetalBuffer result = allocate(backend, 32 * 3 * sizeof(int32_t), "layout check");
  std::memset(scratch.contents(), 0, scratch.sizeBytes());
  Encoder e(backend);
  e.pipeline("layout_check");
  e.bind(scratch, 0);
  e.bind(result, 1);
  e.dispatch(size3(1), size3(32));
  run(backend, e);
  const auto *mismatches = static_cast<const int32_t *>(result.contents());
  int32_t total = 0;
  for (int i = 0; i < 32 * 3; ++i) total += mismatches[i];
  if (total != 0)
    throw metal::MetalBackendError("MPP cooperative-tensor register layout differs from the kernels' slot formula (" +
                                   std::to_string(total) + " mismatches)");
}

void quantize(Encoder &e, const AttentionBuffers &a) {
  const AttentionPrepParams p{a.rows, a.paddedRows, H, device::kKeyBlock, device::kKeyMeanChunks, 1, qkScaleLog2()};
  e.pipeline("dit_attention_key_mean_partial");
  e.bind(a.kHalf, 0);
  e.bind(a.keyMeanPartial, 1);
  e.bytes(p, 2);
  e.dispatch(size3(device::kKeyMeanChunks, H, 1), size3(128, 1, 1));
  e.pipeline("dit_attention_key_mean");
  e.bind(a.keyMeanPartial, 0);
  e.bind(a.keyMean, 1);
  e.bytes(p, 2);
  e.dispatch(size3(H, 1, 1), size3(128, 1, 1));
  e.pipeline("dit_attention_quantize");
  e.bind(a.qHalf, 0);
  e.bind(a.kHalf, 1);
  e.bind(a.qInt8, 2);
  e.bind(a.kInt8, 3);
  e.bind(a.qScale, 4);
  e.bind(a.kScale, 5);
  e.bind(a.keyMean, 6);
  e.bytes(p, 7);
  e.dispatch(size3(a.paddedRows / device::kKeyBlock, H, 2), size3(256, 1, 1));
}

void window(Encoder &e, const AttentionBuffers &a, const WindowTable &table, BufferView out) {
  launches(a, table, 0, [&](const WindowParams &p, int32_t heads) {
    e.pipeline("dit_window_attention");
    e.bind(a.qInt8, 0);
    e.bind(a.kInt8, 1);
    e.bind(a.v, 2);
    e.bind(out, 3);
    e.bind(a.qScale, 4);
    e.bind(a.kScale, 5);
    e.bytes(p, 6);
    e.bind(table.itemBuffer(), 7);
    e.bind(table.tableBuffer(), 8);
    e.bind(a.partials ? a.partials : out, 9);
    e.bind(a.partialStats ? a.partialStats : out, 10);
    e.dispatch(size3(table.items().size() * heads, 1, 1), size3(128, 1, 1));
    if (table.partialSlots() > 0) combine(e, a, table, p, heads, out);
  });
}

void windowQkFix(Encoder &e, const AttentionBuffers &a, BufferView qExact, BufferView kExact, const WindowTable &table,
                 BufferView out) {
  launches(a, table, 0, [&](const WindowParams &p, int32_t heads) {
    e.pipeline("dit_qkfix_window_attention");
    e.bind(a.qHalf, 0);
    e.bind(a.kHalf, 1);
    e.bind(a.v, 2);
    e.bind(out, 3);
    e.bind(qExact, 4);
    e.bind(kExact, 5);
    e.bytes(p, 6);
    e.bind(table.itemBuffer(), 7);
    e.bind(table.tableBuffer(), 8);
    e.bind(a.partials ? a.partials : out, 9);
    e.bind(a.partialStats ? a.partialStats : out, 10);
    e.dispatch(size3(table.items().size() * heads, 1, 1), size3(128, 1, 1));
    if (table.partialSlots() > 0) combine(e, a, table, p, heads, out);
  });
}

} // namespace splash::video::ops
