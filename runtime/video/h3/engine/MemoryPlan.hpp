#pragma once

// Activation memory of one DiT request: every tensor the 50-block forward reads or writes, as a view (offset, bytes)
// of a physical buffer ("slab"), with its lifetime inside a block. Tensors whose lifetimes do not overlap share a slab,
// which is what keeps a 15 s request at ~11 GB of activations (streaming) instead of ~19 GB. A slab is as large as its
// largest view. The block order that makes these aliases safe is engine/Scheduler.mm encodeBlock.
#include "video/h3/engine/Encoder.hpp"
#include "video/h3/model/Geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace h3 {

enum class Slab : uint8_t {
  Residual, Tags, Rope, Text, LinearIn, RowScale, QkHidden, Value, RawQ, RawK,
  Beta, SoftmaxGate, OutGateHidden, FrameMean, Alpha, TextKeys, TextValues, TextA, TextB, TextInverse,
  Pivots, TextPivots, ScanStart, ScanScratch, StateScale, BranchRowScale, Bounds,
  QueryScale, KeyScale, KeyMean, KeyMeanPartial, WindowPartials, WindowStats, QkFixQueryExact, QkFixKeyExact,
  AdaBlocks, AdaFinal, AudioIn, AudioOut, SignsHidden, SignsFfn,
  Count
};

enum class Tensor : uint8_t {
  // residual stream and request inputs
  Residual, Tags, Rope, Text, AdaBlocks, AdaFinal, SignsHidden, SignsFfn,
  VideoIn, VideoOut, AudioIn, AudioOut,
  // linears
  LinearIn, RowScale, Hidden,
  // attention
  QueryHalf, KeyHalf, Value, QueryInt8, KeyInt8, QueryScale, KeyScale, KeyMean, KeyMeanPartial,
  AttentionOut, AttentionOutQkFix, OutProjIn, OutProjInQkFix, WindowPartials, WindowStats,
  // QKFIX
  QkFixInput, QkFixQueryExact, QkFixKeyExact,
  // linear branch
  RawQ, RawK, Beta, SoftmaxGate, OutGateHidden, FrameMean, Alpha, TextKeys, TextValues, KeyFeatures,
  ValueFeatures, StatsA, StatsB, Inverse, TextA, TextB, TextInverse, Pivots, TextPivots, ScanPre, ScanSuf,
  ScanStart, ScanScratch, LinearState, StateScale, Bounds, Readout, BranchIn, BranchRowScale,
  Count
};

struct PlanEntry {
  const char *name;
  Slab slab;
  size_t offset, bytes;
  const char *lifetime;
};

class MemoryPlan {
public:
  // partialSlots: window-attention key-chunk slots per head (0 = none); qkfix: any QKFIX block in the request.
  MemoryPlan(metal::MetalBackend &backend, const model::SequenceLayout &layout, int32_t partialSlots, bool qkfix);

  [[nodiscard]] BufferView view(Tensor t) const;
  [[nodiscard]] const PlanEntry &entry(Tensor t) const { return entries_[static_cast<size_t>(t)]; }
  [[nodiscard]] size_t slabBytes(Slab s) const { return slabBytes_[static_cast<size_t>(s)]; }
  [[nodiscard]] size_t totalBytes() const;

private:
  std::array<PlanEntry, static_cast<size_t>(Tensor::Count)> entries_{};
  std::array<size_t, static_cast<size_t>(Slab::Count)> slabBytes_{};
  std::array<metal::MetalBuffer, static_cast<size_t>(Slab::Count)> slabs_{};
};

} // namespace h3
