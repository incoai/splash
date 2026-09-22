#include "metal/abi/FlashPLEFused.h"
#include <metal_stdlib>

using namespace metal;

// Preserve original q*scale+bias -> BF16 -> shared BF16 scale -> BF16.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

struct FlashPLEFusedSources {
  array<device const uchar *, 128> weights [[id(0)]];
  array<device const uchar *, 128> scales [[id(128)]];
  array<device const uchar *, 128> biases [[id(256)]];
};

inline bool flash_ple_fused_geometry(FlashPLEFusedParams p) {
  return p.lanes && p.rows && p.vocabulary_size &&
         p.eos_token < p.vocabulary_size && p.shard_count == 128 &&
         !p.reserved0 && p.shard_rows && p.table_rows &&
         p.table_rows <= 0x7ffffffffffffffful &&
         p.shard_rows == p.table_rows / 128 && p.table_rows % 128 == 0 &&
         p.weight_row_stride_bytes >= 80 &&
         p.parameter_row_stride_bytes >= 10 &&
         p.parameter_row_stride_bytes % 2 == 0;
}

inline bool flash_ple_fused_token_valid(long token, FlashPLEFusedParams p) {
  return token >= 0 && ulong(token) < p.vocabulary_size;
}

inline long flash_ple_fused_id(
    device const long *tokens, device const long *history,
    device const long *multipliers, device const long *sizes,
    device const long *offsets, device atomic_uint *diagnostics,
    FlashPLEFusedParams p, ulong flat, uint head) {
  const uint lane = uint(flat / p.rows);
  const uint row = uint(flat % p.rows);
  const long current = tokens[flat];
  const long previous = row ? tokens[flat - 1] : history[ulong(lane) * 2 + 1];
  const long older = row > 1 ? tokens[flat - 2] :
                     row == 1 ? history[ulong(lane) * 2 + 1] :
                                history[ulong(lane) * 2];
  // EOS itself retains prior context. The next token resets trigram history.
  const long previous2 = previous == long(p.eos_token) ? long(p.eos_token) : older;
  if (!flash_ple_fused_token_valid(current, p) ||
      !flash_ple_fused_token_valid(previous, p) ||
      !flash_ple_fused_token_valid(previous2, p)) {
    atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
    return -1;
  }
  const long size = sizes[head];
  const long offset = offsets[head];
  if (size <= 0 || offset < 0 || ulong(offset) >= p.table_rows ||
      ulong(size) > p.table_rows - ulong(offset)) {
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return -1;
  }
  // Unsigned multiplication supplies modulo 2^64 wraparound. The resulting
  // bits are interpreted as signed before the positive-modulo correction.
  ulong mixed = ulong(current) * as_type<ulong>(multipliers[0]);
  mixed ^= ulong(previous) * as_type<ulong>(multipliers[1]);
  if (head >= 8)
    mixed ^= ulong(previous2) * as_type<ulong>(multipliers[2]);
  long remainder = as_type<long>(mixed) % size;
  if (remainder < 0) remainder += size;
  return remainder + offset;
}

inline void flash_ple_fused_gather(
    constant FlashPLEFusedSources &source,
    device const bfloat *shared_scale, device bfloat *output,
    device atomic_uint *diagnostics, FlashPLEFusedParams p,
    ulong selection, long source_id, uint column) {
  if (column >= 160) return;
  if (source_id < 0 || ulong(source_id) >= p.table_rows) {
    if (column == 0)
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
    output[selection * 160 + column] = bfloat(as_type<float>(0x7fc00000u));
    return;
  }
  const ulong shard = ulong(source_id) / p.shard_rows;
  const ulong local_row = ulong(source_id) % p.shard_rows;
  device const uchar *packed = source.weights[shard] +
      local_row * p.weight_row_stride_bytes;
  device const bfloat *scales = reinterpret_cast<device const bfloat *>(
      source.scales[shard] + local_row * p.parameter_row_stride_bytes);
  device const bfloat *biases = reinterpret_cast<device const bfloat *>(
      source.biases[shard] + local_row * p.parameter_row_stride_bytes);
  const uint byte = uint(packed[column / 2]);
  const uint code = (byte >> ((column % 2) * 4)) & 15;
  const float product = float(code) * float(scales[column / 32]);
  const bfloat row_value = bfloat(product + float(biases[column / 32]));
  const bfloat value = bfloat(float(row_value) * float(shared_scale[0]));
  if (!metal::isfinite(float(row_value)) || !metal::isfinite(float(value)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  output[selection * 160 + column] = value;
}

// Direct source-ID gather is an independent primitive oracle for the new
// pointer indirection. It also supports source IDs at every shard boundary.
kernel void flash_ple_gather128(
    constant FlashPLEFusedSources &source [[buffer(0)]],
    device const long *ids [[buffer(1)]],
    device const bfloat *shared_scale [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashPLEFusedParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
  if (!flash_ple_fused_geometry(p)) {
    if (column == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const ulong rows = ulong(p.lanes) * p.rows;
  if (group.y >= rows || group.z >= 16) return;
  const ulong selection = ulong(group.y) * 16 + group.z;
  flash_ple_fused_gather(source, shared_scale, output, diagnostics, p,
                         selection, ids[selection], column);
}

// One actual token/head per threadgroup: one exact hash/modulo, then direct
// Q4/G32 gather. IDs are retained for primitive audit and existing scratch
// compatibility; no history writes occur until the following ordered kernel.
kernel void flash_ple_hash_gather128(
    constant FlashPLEFusedSources &source [[buffer(0)]],
    device const long *tokens [[buffer(1)]],
    device const long *history [[buffer(2)]],
    device const long *multipliers [[buffer(3)]],
    device const long *sizes [[buffer(4)]],
    device const long *offsets [[buffer(5)]],
    device const bfloat *shared_scale [[buffer(6)]],
    device long *ids [[buffer(7)]],
    device bfloat *output [[buffer(8)]],
    device atomic_uint *diagnostics [[buffer(9)]],
    constant FlashPLEFusedParams &p [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
  if (!flash_ple_fused_geometry(p)) {
    if (column == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const ulong rows = ulong(p.lanes) * p.rows;
  if (group.y >= rows || group.z >= 16) return;
  const ulong selection = ulong(group.y) * 16 + group.z;
  threadgroup long source_id[1];
  if (column == 0) {
    source_id[0] = flash_ple_fused_id(tokens, history, multipliers, sizes,
                                   offsets, diagnostics, p, group.y, group.z);
    ids[selection] = source_id[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  flash_ple_fused_gather(source, shared_scale, output, diagnostics, p,
                         selection, source_id[0], column);
}
