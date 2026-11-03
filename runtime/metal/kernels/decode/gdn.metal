#include "metal/kernels/common/gdn_decode.h"

// Grid {value heads, lanes}.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, uint RowsInFlight, class Table, class W, class State>
inline void gdn_decode_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device uchar *tape, device const float *a_scale,
    device const bfloat *dt_bias,
    device const W *gdn_norm_weight, device bfloat *gdn_hidden,
    constant GDNDecodeBatchParams &params,
    uint2 group, uint lane, uint simd_group,
    threadgroup GdnDecodeShared<HeadDim> &shared,
    device bfloat *table, device float *sums) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  uint batch = group.y;
  device const uchar *current = SPLASH_LANE_BINDING(batch, current0, current1, current2, current3);
  device uchar *next = SPLASH_LANE_BINDING(batch, next0, next1, next2, next3);
  packed += ulong(batch) * Rows * PackedWidth;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const State *state_in = reinterpret_cast<device const State *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device State *state_out = reinterpret_cast<device State *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);

  const uint pending = params.pending_rows[batch];
  const GdnTape pending_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(tape, params.pending_tape[batch]);
  const GdnTape step_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(tape, params.step_tape[batch]);
  const GdnConvHistory history{conv_state_in, pending_tape.inputs, ConvDim,
                               ConvDim, pending};

  device bfloat *lane_hidden = gdn_hidden + ulong(batch) * Rows * ValueWidth;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      packed, conv_weights, history, conv_state_out, pending_tape, pending,
      step_tape, a_scale, dt_bias, shared, group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gdn_decode_scan<HeadDim, RowsInFlight, State>(state_in, state_out, shared,
                                                pending, group.x, lane,
                                                simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool tiled = (params.flags & GDN_TILED_HEADS) != 0;
  gdn_decode_gate<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      shared, packed, gdn_norm_weight, lane_hidden, tiled,
      (params.flags & GDN_SIGMOID_GATE) != 0, group.x, lane, simd_group);
  if (table) {
    // Each group owns this head for all eight rows; each simdgroup writes
    // the table of the row it just gated.
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint head = gdn_output_head<KeyHeads, ValueHeads>(group.x, tiled);
    for (uint g = 0; g < HeadDim / 64; ++g) {
      const uint column = head * HeadDim + g * 64 + 2 * lane;
      const uint local = simd_group * HeadDim + g * 64 + 2 * lane;
      Table::write(table + ulong(batch) * ValueWidth * Rows,
                   sums + ulong(batch) * Table::sums_per_tile(ValueWidth), ValueWidth,
                   column / 64, simd_group, lane, shared.rows[local], shared.rows[local + 1]);
    }
  }
}

// W: the norm weights' stored type (float: a GGUF's F32 norms, _f32).
#define GDN_DECODE_BUFFERS(W) \
    device const bfloat *packed [[buffer(0)]], \
    device const bfloat *conv_weights [[buffer(1)]], \
    device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]], \
    device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]], \
    device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]], \
    device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]], \
    device uchar *tape [[buffer(10)]], device const float *a_scale [[buffer(11)]], \
    device const bfloat *dt_bias [[buffer(12)]], device const W *gdn_norm_weight [[buffer(13)]], \
    device bfloat *gdn_hidden [[buffer(14)]]
#define GDN_DECODE_THREADS \
    uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, table, sums, Layout, W, State) \
    threadgroup GdnDecodeShared<HeadDim> shared; \
    gdn_decode_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, 2, Layout, W, State>( \
        packed, conv_weights, current0, current1, current2, current3, next0, \
        next1, next2, next3, tape, a_scale, dt_bias, \
        gdn_norm_weight, gdn_hidden, params, group, lane, simd_group, shared, \
        table, sums);
// Entries without a table pass null pointers, which skip the write; their Layout only completes the template.
#define GDN_DECODE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, W, State) \
  kernel void Name(GDN_DECODE_BUFFERS(W), \
      constant GDNDecodeBatchParams &params [[buffer(15)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, nullptr, nullptr, gguf_sg::Table16, W, State) \
  }
// The out-projection's table (Layout: gguf_sg::Table16).
#define GDN_DECODE_TABLE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Layout, W, State) \
  kernel void Name(GDN_DECODE_BUFFERS(W), \
      device bfloat *table [[buffer(15)]], device float *sums [[buffer(16)]], \
      constant GDNDecodeBatchParams &params [[buffer(17)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, table, sums, Layout, W, State) \
  }
// Each entry over an fp32 state and, as Name_bf16_state, a bf16 one.
#define GDN_DECODE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, W) \
  GDN_DECODE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, W, float) \
  GDN_DECODE_KERNEL(Name##_bf16_state, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, W, bfloat)
#define GDN_DECODE_TABLE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Layout, W) \
  GDN_DECODE_TABLE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Layout, W, float) \
  GDN_DECODE_TABLE_KERNEL(Name##_bf16_state, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Layout, W, bfloat)

// Two rows overlap reductions and arithmetic without the register cost of four.
GDN_DECODE_ENTRY(verify_gdn_fused, 16, 48, 128, 10240, 16640, bfloat)
GDN_DECODE_ENTRY(verify_gdn_fused_vh32, 16, 32, 128, 8192, 12544, bfloat)
// Table16 feeds the register tile, after norms that are F32 (a GGUF's) or bf16 (an MLX target's).
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16, 16, 48, 128, 10240, 16640, gguf_sg::Table16, bfloat)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh32, 16, 32, 128, 8192, 12544, gguf_sg::Table16, bfloat)
GDN_DECODE_ENTRY(verify_gdn_fused_f32, 16, 48, 128, 10240, 16640, float)
GDN_DECODE_ENTRY(verify_gdn_fused_vh32_f32, 16, 32, 128, 8192, 12544, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_f32, 16, 48, 128, 10240, 16640, gguf_sg::Table16, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh32_f32, 16, 32, 128, 8192, 12544, gguf_sg::Table16, float)
// Qwen3.5-122B-A10B (64 value heads).
GDN_DECODE_ENTRY(verify_gdn_fused_vh64, 16, 64, 128, 12288, 20736, bfloat)
GDN_DECODE_ENTRY(verify_gdn_fused_vh64_f32, 16, 64, 128, 12288, 20736, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh64, 16, 64, 128, 12288, 20736, gguf_sg::Table16, bfloat)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh64_f32, 16, 64, 128, 12288, 20736, gguf_sg::Table16, float)
// Qwen3.8-Flash-Next (16512-column rows).
GDN_DECODE_ENTRY(verify_gdn_fused_p16512, 16, 48, 128, 10240, 16512, bfloat)
GDN_DECODE_ENTRY(verify_gdn_fused_p16512_f32, 16, 48, 128, 10240, 16512, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_p16512, 16, 48, 128, 10240, 16512, gguf_sg::Table16, bfloat)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_p16512_f32, 16, 48, 128, 10240, 16512, gguf_sg::Table16, float)
#undef GDN_DECODE_ENTRY
#undef GDN_DECODE_TABLE_ENTRY
#undef GDN_DECODE_KERNEL
#undef GDN_DECODE_TABLE_KERNEL
#undef GDN_DECODE_BODY
#undef GDN_DECODE_THREADS
#undef GDN_DECODE_BUFFERS
