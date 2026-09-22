#include "metal/abi/FlashPLE.h"
#include <metal_stdlib>

using namespace metal;

// Array-operation boundaries in the canonical implementation are BF16. The
// hash is instead an I64 wraparound computation, including signed modulo.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline float flash_ple_nan() { return as_type<float>(0x7fc00000u); }

inline bool flash_ple_hash_geometry(FlashPLEHashParams p) {
  return p.lanes && p.rows && p.heads_per_ngram == 8 &&
         p.vocabulary_size && p.eos_token < p.vocabulary_size &&
         !p.reserved0 && p.table_rows && p.table_rows <= 0x7ffffffffffffffful;
}

inline bool flash_ple_token_valid(long token, FlashPLEHashParams p) {
  return token >= 0 && ulong(token) < p.vocabulary_size;
}

// 0 I64[lanes,rows] tokens,1 I64[lanes,2] history,2 stored I64[3] multipliers,
// 3 stored I64[16] sizes,4 stored I64[16] offsets,5 I64[lanes,rows,16] IDs,
// 6 sticky diagnostics,7 params. The history is read-only in this dispatch.
kernel void flash_ple_hash(
    device const long *tokens [[buffer(0)]],
    device const long *history [[buffer(1)]],
    device const long *multipliers [[buffer(2)]],
    device const long *sizes [[buffer(3)]],
    device const long *offsets [[buffer(4)]],
    device long *ids [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashPLEHashParams &p [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (!flash_ple_hash_geometry(p)) {
    if (index == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const ulong count = ulong(p.lanes) * p.rows * 16;
  if (index >= count) return;
  const uint head = uint(index % 16);
  const ulong flat = index / 16;
  const uint lane = uint(flat / p.rows);
  const uint row = uint(flat % p.rows);
  const long current = tokens[flat];
  const long previous = row ? tokens[flat - 1] : history[ulong(lane) * 2 + 1];
  const long older = row > 1 ? tokens[flat - 2] :
                     row == 1 ? history[ulong(lane) * 2 + 1] :
                                history[ulong(lane) * 2];
  const long previous2 = previous == long(p.eos_token) ? long(p.eos_token) : older;
  if (!flash_ple_token_valid(current, p) ||
      !flash_ple_token_valid(previous, p) ||
      !flash_ple_token_valid(previous2, p)) {
    atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
    ids[index] = -1;
    return;
  }
  const long size = sizes[head];
  const long offset = offsets[head];
  if (size <= 0 || offset < 0 || ulong(offset) >= p.table_rows ||
      ulong(size) > p.table_rows - ulong(offset)) {
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    ids[index] = -1;
    return;
  }
  ulong mixed = ulong(current) * as_type<ulong>(multipliers[0]);
  mixed ^= ulong(previous) * as_type<ulong>(multipliers[1]);
  if (head >= 8)
    mixed ^= ulong(previous2) * as_type<ulong>(multipliers[2]);
  long remainder = as_type<long>(mixed) % size;
  if (remainder < 0) remainder += size;
  ids[index] = remainder + offset;
}

kernel void flash_ple_update_history(
    device const long *tokens [[buffer(0)]],
    device long *history [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashPLEHashParams &p [[buffer(3)]],
    uint lane [[thread_position_in_grid]]) {
  if (!flash_ple_hash_geometry(p)) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (lane >= p.lanes) return;
  for (uint row = 0; row < p.rows; ++row)
    if (!flash_ple_token_valid(tokens[lane * p.rows + row], p)) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      return;
    }
  const long older = p.rows > 1 ? tokens[lane * p.rows + p.rows - 2] :
                                 history[lane * 2 + 1];
  const long previous = tokens[lane * p.rows + p.rows - 1];
  history[lane * 2] = older;
  history[lane * 2 + 1] = previous;
}

#define FLASH_PLE_SHARD_ARGS(S, B)                                            \
  device const uchar *weights##S [[buffer(B)]],                              \
  device const uchar *scales##S [[buffer(B + 1)]],                             \
  device const uchar *biases##S [[buffer(B + 2)]]

// 0 IDs,1 checkpoint BF16 shared scale,2 BF16 output,3 diagnostics,
// 4..27 eight W/S/B triples,28 params. Every triple keeps a source shard alive.
kernel void flash_ple_gather8(
    device const long *ids [[buffer(0)]],
    device const bfloat *shared_scale [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    FLASH_PLE_SHARD_ARGS(0, 4), FLASH_PLE_SHARD_ARGS(1, 7),
    FLASH_PLE_SHARD_ARGS(2, 10), FLASH_PLE_SHARD_ARGS(3, 13),
    FLASH_PLE_SHARD_ARGS(4, 16), FLASH_PLE_SHARD_ARGS(5, 19),
    FLASH_PLE_SHARD_ARGS(6, 22), FLASH_PLE_SHARD_ARGS(7, 25),
    constant FlashPLEGatherParams &p [[buffer(28)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
  if (!p.rows || p.heads != 16 || p.head_width != 160 ||
      !p.shard_count || p.shard_count > 8 || !p.shard_rows ||
      !p.table_rows || p.table_rows > 0x7ffffffffffffffful ||
      p.first_row >= p.table_rows ||
      p.shard_count > (p.table_rows - p.first_row) / p.shard_rows ||
      p.weight_row_stride_bytes < 80 ||
      p.parameter_row_stride_bytes < 10 ||
      p.parameter_row_stride_bytes % 2) {
    if (column == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (group.y >= p.rows || group.z >= p.heads || column >= p.head_width)
    return;
  const ulong selection = ulong(group.y) * p.heads + group.z;
  const long source_id = ids[selection];
  if (source_id < 0 || ulong(source_id) >= p.table_rows) {
    if (p.first_row == 0) {
      if (column == 0)
        atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      output[selection * p.head_width + column] = bfloat(flash_ple_nan());
    }
    return;
  }
  if (ulong(source_id) < p.first_row) return;
  const ulong relative = ulong(source_id) - p.first_row;
  const ulong slot = relative / p.shard_rows;
  if (slot >= p.shard_count) return;
  const ulong local_row = relative % p.shard_rows;
  device const uchar *weights = weights0;
  device const uchar *scales = scales0;
  device const uchar *biases = biases0;
#define FLASH_PLE_SELECT(S)                                                  \
  if (slot == S) { weights = weights##S; scales = scales##S; biases = biases##S; }
  FLASH_PLE_SELECT(1) FLASH_PLE_SELECT(2) FLASH_PLE_SELECT(3)
  FLASH_PLE_SELECT(4) FLASH_PLE_SELECT(5) FLASH_PLE_SELECT(6)
  FLASH_PLE_SELECT(7)
#undef FLASH_PLE_SELECT
  const device uchar *packed = weights + local_row * p.weight_row_stride_bytes;
  const device bfloat *s = reinterpret_cast<const device bfloat *>(
      scales + local_row * p.parameter_row_stride_bytes);
  const device bfloat *b = reinterpret_cast<const device bfloat *>(
      biases + local_row * p.parameter_row_stride_bytes);
  const uint byte = uint(packed[column / 2]);
  const uint code = (byte >> ((column % 2) * 4)) & 15;
  const float product = float(code) * float(s[column / 32]);
  const bfloat row_value = bfloat(product + float(b[column / 32]));
  const bfloat value = bfloat(float(row_value) * float(shared_scale[0]));
  if (!metal::isfinite(float(row_value)) || !metal::isfinite(float(value)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  output[selection * p.head_width + column] = value;
}

#undef FLASH_PLE_SHARD_ARGS

inline bool flash_ple_post_geometry(FlashPLEPostParams p) {
  return p.lanes && p.rows && p.width && p.streams && p.streams <= 8 &&
         !(p.flags & ~1u) && p.state_rows == 9 && p.taps == 4 && p.dilation == 3;
}

inline bfloat flash_ple_sigmoid(bfloat input) {
  // This helper supplies compiled nn.silu in the convolution epilogue.
  // Preserve the actual MLX Metal functor's input type and promotions. In
  // particular exp(abs(BF16)) must not become exp(abs(F32)) before the result.
  // MLX supplies a BF16 exp overload. Spell out that cast without importing
  // its headers; integer/BF16 arithmetic itself returns BF16 in Metal 4.1.
  const bfloat exponential = bfloat(metal::exp(metal::abs(float(input))));
  auto tail = 1 / (1 + exponential);
  return input < 0 ? tail : 1 - tail;
}

inline bfloat flash_ple_gate_sigmoid(bfloat input) {
  // The gate calls standalone mx.sigmoid, whose unary GPU kernel uses
  // precise exp. Compiled nn.silu below retains the fast-exp helper instead.
  const bfloat exponential =
      bfloat(metal::precise::exp(metal::abs(float(input))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  return input < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

// keys and queries already normalized by the canonical grouped RMSNorm.
// The gate's BF16 array operations are retained between nonlinear functions.
kernel void flash_ple_gate(
    device const bfloat *keys [[buffer(0)]],
    device const bfloat *queries [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const uint *mask [[buffer(3)]],
    device bfloat *gated [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashPLEPostParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]) {
  if (!flash_ple_post_geometry(p)) {
    if (thread_index == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (ulong(group.x) >= ulong(p.lanes) * p.rows || group.y >= p.streams) return;
  const ulong base = (ulong(group.x) * p.streams + group.y) * p.width;
  const uint threads = group_size.x;
  threadgroup bfloat partials[32];
  threadgroup bfloat gate;
  bfloat reduced = bfloat(0.0f);
  if (p.width <= 64) {
    // MLX's small-row reduction has a BF16 accumulator, one sequential fold
    // per output. Widening this sum to F32 changes source gating substantially.
    if (thread_index == 0)
      for (uint channel = 0; channel < p.width; ++channel) {
        const bfloat product = bfloat(float(keys[base + channel]) *
                                      float(queries[base + channel]));
        reduced = bfloat(float(product) + float(reduced));
      }
  } else {
    // Four consecutive columns per thread, BF16 after every local addition.
    // H=2560 uses 640 threads and exactly one such four-column block each.
    bfloat total = bfloat(0.0f);
    for (ulong block = 0; block < p.width; block += ulong(threads) * 4) {
      for (uint read = 0; read < 4; ++read) {
        const ulong channel = block + ulong(thread_index) * 4 + read;
        if (channel >= p.width) break;
        const bfloat product = bfloat(float(keys[base + channel]) *
                                      float(queries[base + channel]));
        total = bfloat(float(product) + float(total));
      }
    }
    const bfloat simd_total = bfloat(simd_sum(float(total)));
    if (lane == 0) partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint simd_count = (threads + 31) / 32;
    const bfloat partial = thread_index < simd_count ?
        partials[thread_index] : bfloat(0.0f);
    // MLX's BF16 simd_sum overload uses F32 within the SIMD reduction, then
    // rounds to BF16. Both SIMD levels retain that same boundary.
    reduced = bfloat(simd_sum(float(partial)));
  }
  if (thread_index == 0) {
    const bfloat divisor = bfloat(metal::sqrt(float(p.width)));
    const bfloat divided = bfloat(float(reduced) / float(divisor));
    const bfloat magnitude = bfloat(metal::max(metal::abs(float(divided)),
                                               float(bfloat(1e-6f))));
    const bfloat root = bfloat(metal::sqrt(float(magnitude)));
    const float sign = float(divided) < 0 ? -1.0f :
                       float(divided) > 0 ? 1.0f : 0.0f;
    gate = flash_ple_gate_sigmoid(bfloat(sign * float(root)));
    if (!metal::isfinite(float(reduced)) ||
        !metal::isfinite(float(divided)) || !metal::isfinite(float(root)) ||
        !metal::isfinite(float(gate)))
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool enabled = !(p.flags & 1u) || mask[group.x] != 0;
  for (uint channel = thread_index; channel < p.width; channel += threads) {
    const bfloat value = bfloat(float(gate) *
        float(values[ulong(group.x) * p.width + channel]));
    if (!metal::isfinite(float(value)))
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    gated[base + channel] = enabled ? value : bfloat(0.0f);
  }
}

kernel void flash_ple_convolution(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *gated [[buffer(1)]],
    device const bfloat *state [[buffer(2)]],
    device const bfloat *weights [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashPLEPostParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (!flash_ple_post_geometry(p)) {
    if (thread_index == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint hyper_width = p.width * p.streams;
  const uint channel = group.x * 256 + thread_index;
  if (channel >= hyper_width || ulong(group.y) >= ulong(p.lanes) * p.rows) return;
  const uint lane = group.y / p.rows;
  const uint row = group.y % p.rows;
  float sum = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const uint timeline = row + tap * 3;
    const bfloat value = timeline < 9 ?
        state[(ulong(lane) * 9 + timeline) * hyper_width + channel] :
        normalized[(ulong(lane) * p.rows + timeline - 9) * hyper_width + channel];
    sum += float(value) * float(weights[ulong(channel) * 4 + tap]);
  }
  const bfloat convolution = bfloat(sum);
  const bfloat sigmoid = flash_ple_sigmoid(convolution);
  const bfloat activated = bfloat(float(convolution) * float(sigmoid));
  const ulong out = ulong(group.y) * hyper_width + channel;
  const bfloat value = bfloat(float(gated[out]) + float(activated));
  if (!metal::isfinite(float(value)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  output[out] = value;
}

// Each thread owns one channel and walks old state in chronological order.
// When rows<9 all source state indices are strictly later than the destination,
// so exact in-place state updates cannot race or overwrite an unread value.
kernel void flash_ple_update_convolution_state(
    device const bfloat *normalized [[buffer(0)]],
    device bfloat *state [[buffer(1)]],
    constant FlashPLEPostParams &p [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (!flash_ple_post_geometry(p)) return;
  const uint hyper_width = p.width * p.streams;
  const uint channel = group.x * 256 + thread_index;
  if (channel >= hyper_width || group.y >= p.lanes) return;
  for (uint row = 0; row < 9; ++row) {
    const ulong timeline = ulong(p.rows) + row;
    const bfloat value = timeline < 9 ?
        state[(ulong(group.y) * 9 + timeline) * hyper_width + channel] :
        normalized[(ulong(group.y) * p.rows + timeline - 9) * hyper_width + channel];
    state[(ulong(group.y) * 9 + row) * hyper_width + channel] = value;
  }
}

kernel void flash_ple_inject(
    device const bfloat *hyper [[buffer(0)]],
    device const bfloat *ple [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant FlashPLEPostParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (!flash_ple_post_geometry(p)) return;
  const uint hyper_width = p.width * p.streams;
  const uint channel = group.x * 256 + thread_index;
  if (channel >= hyper_width || ulong(group.y) >= ulong(p.lanes) * p.rows) return;
  const ulong index = ulong(group.y) * hyper_width + channel;
  output[index] = bfloat(float(hyper[index]) + float(ple[index]));
}

kernel void flash_ple_restore_prefix(
    device const long *before_history [[buffer(0)]],
    device const long *tokens [[buffer(1)]],
    device const bfloat *before_state [[buffer(2)]],
    device const bfloat *normalized [[buffer(3)]],
    device const uint *kept_tokens [[buffer(4)]],
    device long *output_history [[buffer(5)]],
    device bfloat *output_state [[buffer(6)]],
    device atomic_uint *diagnostics [[buffer(7)]],
    constant FlashPLEPrefixParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const FlashPLEPostParams p = params.geometry;
  if (!flash_ple_post_geometry(p) || !params.vocabulary_size ||
      params.reserved0 || params.reserved1 || params.reserved2) {
    if (thread_index == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint hyper_width = p.width * p.streams;
  const uint channel = group.x * 256 + thread_index;
  if (channel >= hyper_width || group.y >= p.lanes) return;
  const uint kept = kept_tokens[group.y];
  if (kept > p.rows) {
    if (channel == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  for (uint row = 0; row < 9; ++row) {
    const ulong timeline = ulong(kept) + row;
    const bfloat value = timeline < 9 ?
        before_state[(ulong(group.y) * 9 + timeline) * hyper_width + channel] :
        normalized[(ulong(group.y) * p.rows + timeline - 9) * hyper_width + channel];
    output_state[(ulong(group.y) * 9 + row) * hyper_width + channel] = value;
  }
  if (channel == 0) {
    for (uint row = 0; row < kept; ++row) {
      const long token = tokens[ulong(group.y) * p.rows + row];
      if (token < 0 || ulong(token) >= params.vocabulary_size) {
        atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
        return;
      }
    }
    for (uint slot = 0; slot < 2; ++slot) {
      const ulong timeline = ulong(kept) + slot;
      output_history[ulong(group.y) * 2 + slot] = timeline < 2 ?
          before_history[ulong(group.y) * 2 + timeline] :
          tokens[ulong(group.y) * p.rows + timeline - 2];
    }
  }
}
