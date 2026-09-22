#pragma once

#include "FlashGDN.hpp"

#include <cstdint>
#include <span>

namespace splash::flash {

class FlashGDNLazyRollback;

inline constexpr uint64_t kFlashBatchVerifyGDNConvolutionRowStrideBytes = 65536;
inline constexpr uint64_t kFlashBatchVerifyGDNRecurrentRowStrideBytes =
    flashGDNRecurrentLaneBytes();

// Exact provisional GDN for 1..4 real lanes and 1..4 rows per lane. Inputs
// and outputs retain the GDN lane-major [lane,row,width] ABI. Each incoming
// request state describes one lane; request buffers remain unchanged until
// all shorter convolution/recurrent prefixes have been recorded.
//
// packedState uses the row strides above as its lane strides; zero stride
// selects those defaults. Prefix tapes describe one layer in
// [lane,maximumRows-1,padded state] order, including unused configured rows.
// A prefix at index q-1 records state after consuming q rows. The full final
// prefix is scattered to the request states and omitted from the tapes.
// Tapes are unused and may be empty for rows=1. An optional lazy arena saves
// initial state and prepared operands instead of per-row prefixes. It owns a
// pending ticket until its caller commits or terminally aborts the trial.
//
// Work, request, packed-state, and active tape buffers must be Shared so host
// validation can reject partial aliases before adding graph dispatches.
// Immutable weights may use either storage. No device contents are read.
void addBatchVerifyGDN(
    metal::CommandGraph &graph, metal::MetalBackend &backend,
    const FlashGDNWeights &weights, const FlashGDNBuffers &buffers,
    std::span<const FlashGDNState> actualRequestStates,
    const FlashGDNState &packedState,
    const metal::MetalBuffer &convolutionPrefixes,
    const metal::MetalBuffer &recurrentPrefixes, uint32_t rows,
    uint32_t maximumRows = 4, float normEpsilon = 1e-6f,
    FlashGDNLazyRollback *lazy = nullptr, uint64_t *lazyTicket = nullptr);

} // namespace splash::flash
