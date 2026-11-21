#pragma once

#include "ops/AneFfn.hpp"
#include "ops/AneFfnCalibration.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace splash::ops::ane_ffn {

// Times the prefill FFN's Neural Engine split of `layers` (AneFfn) on this
// device, the GPU's part through `linear`, with chunks in `ffn` and the
// residual and output of alternate layers in `hidden`, all of which it
// overwrites, through the split's path: begin(), its layers as add() encodes
// them, commit() and its completion check (layer() in
// AneFfnMeasurement.cpp). A program it compiles waits for the ANE's service
// until `interrupted` returns true. Each duration passes usable() where it is
// measured.
class Measurement final {
public:
  // A split of `slots` slots, which pipelines as many chunks of a command,
  // is also timed so (Timing::ane), and `mixers`, the layers' mixer
  // projections (one a layer), stand in for their mixers in time().
  Measurement(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
              const PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden,
              std::function<bool()> interrupted, uint32_t slots = 1, std::span<const MixerProjections> mixers = {});

  // The split's GPU part alone and its split layer of full chunks with the
  // ANE taking the channel units nearest two shares, the GPU's FFN alone and
  // the mixers' projections, on the same four layers spread over the model,
  // whose formats can differ with depth: a program each, which the ANE's
  // service keeps once compiled.
  [[nodiscard]] Timings time();
  // The same at `aneUnits`, the split layer less `gpuAlone`, the GPU's FFN
  // alone that time() timed: what calibration fits again at the units it
  // first chooses (ane_ffn::refit), on the layers it fitted.
  [[nodiscard]] Timing time(uint32_t aneUnits, double gpuAlone);
  // The GPU alone over the least and the most rows the split takes, and the
  // split with the ANE taking `aneUnits` over a chunk of each count of its
  // pieces, each chunk alone, on the layers time() times: the program the
  // split that serves compiles.
  [[nodiscard]] ChunkTimings chunks(uint32_t aneUnits);

private:
  // The per-layer milliseconds of `count` `rows`-row FFN layers of `layers`
  // from the first, fewer than all of them (AneFfnMeasurement.cpp): the
  // GPU's alone without `split`, or `split`'s less `standIn`; and of
  // `split`'s GPU part alone of full chunks.
  [[nodiscard]] double layer(std::span<const SwiGluProjections> layers, uint32_t count, uint32_t rows,
                             AneFfn *split = nullptr, double standIn = 0.0);
  [[nodiscard]] double gpuPart(std::span<const SwiGluProjections> layers, AneFfn &split);
  // The ANE's milliseconds a full chunk's layer of `split`, every slot's
  // chunk of each of all but the last of `layers` started before any joins
  // (AneFfnMeasurement.cpp).
  [[nodiscard]] double pipelinedAne(std::span<const SwiGluProjections> layers, AneFfn &split);
  // The kSampledLayers layers time() spreads over the model, first and last
  // among them (AneFfnMeasurement.cpp).
  [[nodiscard]] std::vector<SwiGluProjections> sampled() const;
  // The per-layer milliseconds of the sampled layers' mixers on full chunks,
  // alone on the GPU, as their projections give them (AneFfnMeasurement.cpp);
  // 0 without mixers.
  [[nodiscard]] double mixer();

  metal::MetalBackend &backend_;
  std::span<const SwiGluProjections> layers_;
  PrefillFfnBuffers ffn_;
  std::array<metal::MetalBuffer, 2> hidden_;
  std::function<bool()> interrupted_;
  std::span<const MixerProjections> mixers_;
  uint32_t slots_ = 1;
  const Linear linear_;
};

} // namespace splash::ops::ane_ffn
