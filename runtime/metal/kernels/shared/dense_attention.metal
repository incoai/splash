// Flash attention of a sequence over its own keys (ops/DenseAttention.hpp):
// every head its own keys and values, each row every key of the sequence
// (metal/abi/DenseAttention.h). Grid: {ceil(rows / 32), heads} of 128
// threads, the query rows and threads of metal/abi/DenseAttention.h.
// Queries and keys hold each head's D dimensions padded with zeros to DQ
// (splash_dense_attention_padded). A threadgroup reads whole 128-key tiles
// up to the one that holds the sequence's last key, and masks the keys past
// it.
//
// A threadgroup takes 32 query rows on four simdgroups and 24.25 KiB of
// threadgroup memory, which keeps several resident per core. Scores, bf16
// probabilities and the row statistics each have their own storage, so a
// key tile needs two barriers: one before the softmax reads its scores and
// one before PV reads its probabilities. The next tile stores its scores
// past the second, and writes its probabilities past its first, which no
// thread reaches before finishing this tile's PV. PV accumulates into the
// running output, which each tile first rescales by its rows' maximum
// change, so one M x D fp32 cooperative tensor is live.

#include "metal/abi/DenseAttention.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// dense_attention_{D}: heads of D dimensions.
template <ushort D>
kernel void dense_attention(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant DenseAttentionParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    constexpr ushort M = SPLASH_DENSE_ATTENTION_ROWS,
                     N = SPLASH_DENSE_ATTENTION_KEYS;
    constexpr ushort DQ = splash_dense_attention_padded(D);
    constexpr ushort Simdgroups = SPLASH_DENSE_ATTENTION_THREADS / 32;
    // Four lanes own one row, four consecutive keys of every 16, so the row
    // maximum and sum are two xor shuffles and scores and probabilities move
    // as vectors.
    constexpr ushort LanesPerRow = Simdgroups * 32 / M;
    constexpr ushort Runs = N / (LanesPerRow * 4);
    static_assert(Simdgroups * 32 == SPLASH_DENSE_ATTENTION_THREADS &&
                      LanesPerRow * M == Simdgroups * 32,
                  "whole simdgroups whose lanes own whole rows");
    static_assert(LanesPerRow == 4, "two xor shuffles reduce a row");
    alignas(16) threadgroup float score_storage[M * N];
    alignas(16) threadgroup bfloat probabilities[M * N];
    threadgroup float row_sum[M];
    threadgroup float previous_scale[M];
    const ushort row = thread_index / LanesPerRow;
    const ushort part = thread_index % LanesPerRow;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const uint head = group.y;
    const uint query_start = group.x * M;
    const uint last_row = min(query_start + M, params.rows) - 1;
    const uint end_tile = (params.rows + N - 1) / N;
    const int q_stride = int(params.query_row_stride);
    const int k_stride = int(params.key_row_stride);
    const int v_stride = int(params.value_row_stride);
    device bfloat *q = queries + ulong(head) * params.query_head_stride +
        ulong(query_start) * params.query_row_stride;
    device bfloat *k = keys + ulong(head) * params.key_head_stride;
    device bfloat *v = values + ulong(head) * params.value_head_stride;
    device bfloat *out = output + ulong(head) * params.output_head_stride +
        ulong(query_start) * params.output_row_stride;

    auto qt = tensor(q, dextents<int, 2>{DQ, M}, array<int, 2>{1, q_stride});
    constexpr auto qk_descriptor =
        matmul2d_descriptor(M, N, DQ, false, true, false);
    matmul2d<qk_descriptor, execution_simdgroups<Simdgroups>> qk;
    auto q0 = qt.template slice<DQ, M>(0, 0);

    auto pt = tensor(probabilities, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto p0 = pt.template slice<N, M>(0, 0);
    auto st = tensor(score_storage, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto s0 = st.template slice<N, M>(0, 0);
    auto first_v = tensor(v, dextents<int, 2>{D, N},
                          array<int, 2>{1, v_stride});
    auto first_v0 = first_v.template slice<D, N>(0, 0);
    constexpr auto pv_descriptor = matmul2d_descriptor(
        M, D, N, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_descriptor, execution_simdgroups<Simdgroups>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), float>();
    // Every loop over the output touches its valid elements alone: a
    // padding element's coordinates need not name a row of the statistics
    // (Apple10 lays 72 columns out over 96).
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (running.is_valid_element(i))
            running[i] = 0.0f;
    }
    threadgroup const float4 *row_scores =
        reinterpret_cast<threadgroup const float4 *>(score_storage + row * N);
    threadgroup bfloat4 *row_probabilities =
        reinterpret_cast<threadgroup bfloat4 *>(probabilities + row * N);
    for (uint tile = 0; tile < end_tile; ++tile) {
        device bfloat *tile_k = k + ulong(tile) * N * k_stride;
        auto kt = tensor(tile_k, dextents<int, 2>{DQ, N},
                         array<int, 2>{1, k_stride});
        auto k0 = kt.template slice<DQ, N>(0, 0);
        auto scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(k0), float>();
        qk.run(q0, k0, scores);
        scores.store(s0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Keys at or past `limit` lie past the sequence and are masked.
        const int limit = int(params.rows) - int(tile * N);
        float logits[Runs * 4];
        float tile_max = -INFINITY;
        for (ushort r = 0; r < Runs; ++r) {
            const ushort run = r * LanesPerRow + part;
            const float4 score = row_scores[run] * params.scale;
            for (ushort j = 0; j < 4; ++j) {
                const int key = int(run * 4 + j);
                logits[r * 4 + j] = key < limit ? score[j] : -INFINITY;
                tile_max = max(tile_max, logits[r * 4 + j]);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 2));
        // Key 0 of every tile lies below `limit`, so each row's maximum is
        // finite from the first tile on.
        const float next_max = max(running_max, tile_max);
        const float scale = fast::exp(running_max - next_max);
        float tile_sum = 0.0f;
        for (ushort r = 0; r < Runs; ++r) {
            float4 exponents;
            for (ushort j = 0; j < 4; ++j) {
                exponents[j] = fast::exp(logits[r * 4 + j] - next_max);
                tile_sum += exponents[j];
            }
            row_probabilities[r * LanesPerRow + part] = bfloat4(exponents);
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 2);
        running_sum = running_sum * scale + tile_sum;
        running_max = next_max;
        if (part == 0) {
            previous_scale[row] = scale;
            row_sum[row] = running_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort i = 0; i < running.get_capacity(); ++i) {
            if (!running.is_valid_element(i))
                continue;
            auto index = running.get_multidimensional_index(i);
            running[i] *= previous_scale[index[1]];
        }
        device bfloat *tile_v = v + ulong(tile) * N * v_stride;
        auto vt = tensor(tile_v, dextents<int, 2>{D, N},
                         array<int, 2>{1, v_stride});
        auto v0 = vt.template slice<D, N>(0, 0);
        pv.run(p0, v0, running);
    }

    // The rows of the sequence, normalized, stored as one bf16 tile whose
    // extent ends at the last row, so rows past it are not stored: element
    // by element, the stores took 5% more of an M5 Max's attention of 1536
    // patches.
    auto converted = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), bfloat>();
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running.is_valid_element(i))
            continue;
        auto index = running.get_multidimensional_index(i);
        converted[i] = bfloat(running[i] / row_sum[index[1]]);
    }
    auto ot = tensor(out, dextents<int, 2>{D, int(last_row + 1 - query_start)},
                     array<int, 2>{1, int(params.output_row_stride)});
    converted.store(ot);
}

#define DENSE_ATTENTION_KERNEL(dimensions)                                    \
template [[host_name("dense_attention_" #dimensions)]] kernel void            \
dense_attention<dimensions>(device bfloat *, device bfloat *, device bfloat *, \
                            device bfloat *, constant DenseAttentionParams &, \
                            uint2, uint);

// The vision tower's heads of 72.
DENSE_ATTENTION_KERNEL(72)
