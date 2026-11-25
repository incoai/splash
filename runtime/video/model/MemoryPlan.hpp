#pragma once

// Activation memory of one DiT request: every tensor the 50-block forward reads or writes, as a view (offset, bytes)
// of a physical buffer ("slab"), with its lifetime inside a block. Tensors whose lifetimes do not overlap share a slab,
// which is what keeps a 15 s request at ~11 GB of activations (streaming) instead of ~19 GB. A slab is as large as its
// largest view. The block order that makes these aliases safe is stages/DenoisingStage.mm encodeBlock.
#include "video/metal/Encoder.hpp"
#include "video/ops/Geometry.hpp"
#include "video/ops/VsaTable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace splash::video::model {

enum class Slab : uint8_t {
  Residual, Tags, Rope, Text, LinearIn, RowScale, QkHidden, Value, RawQ, RawK,
  Beta, SoftmaxGate, OutGateHidden, FrameMean, Alpha, TextKeys, TextValues, TextA, TextB, TextInverse,
  Pivots, TextPivots, ScanStart, ScanScratch, StateScale, BranchRowScale, Bounds,
  QueryScale, KeyScale, KeyMean, KeyMeanPartial, WindowPartials, WindowStats, QkFixQueryExact, QkFixKeyExact,
  AdaBlocks, AdaFinal, AudioIn, AudioOut, SignsHidden, SignsFfn,
  VsaPooled, VsaOutC, VsaSelected, TagsPacked,
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
  // FastH3 (VSA)
  Gate, PooledQ, PooledK, PooledV, OutC, Selected, ResidualPacked, TagsPacked, VsaScores,
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
  // partialSlots: window-attention key-chunk slots per head (0 = none); vsa: the FastH3 tile map (VSA blocks: no
  // linear branch, rows in tile order), null for VDN-H3's window softmax.
  MemoryPlan(metal::MetalBackend &backend, const ops::SequenceLayout &layout, int32_t partialSlots,
             const ops::VsaTable *vsa = nullptr);

  [[nodiscard]] BufferView view(Tensor t) const;
  [[nodiscard]] size_t totalBytes() const;
  // totalBytes() of the plan of a request, without allocating it: every slab scales with the request's rows (the
  // canvas's rows per latent frame x latent frames, text, audio), FastH3's also with its tiles.
  [[nodiscard]] static size_t bytesFor(const ops::SequenceLayout &layout, int32_t partialSlots,
                                       const ops::VsaTable *vsa = nullptr);

private:
  using Entries = std::array<PlanEntry, static_cast<size_t>(Tensor::Count)>;
  using SlabBytes = std::array<size_t, static_cast<size_t>(Slab::Count)>;
  static void layOut(const ops::SequenceLayout &layout, int32_t partialSlots, const ops::VsaTable *vsa,
                     Entries &entries, SlabBytes &slabBytes);

  Entries entries_{};
  SlabBytes slabBytes_{};
  std::array<metal::MetalBuffer, static_cast<size_t>(Slab::Count)> slabs_{};
};

} // namespace splash::video::model
