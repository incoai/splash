#pragma once

#include "ops/AneFfn.hpp"
#include "ops/AneFfnCalibration.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>

namespace splash::ops::ane_ffn {

// Times the prefill FFN's Neural Engine split of `layers` (AneFfn) on this
// device, with chunks in `ffn` and the residual and output of alternate
// layers in `hidden`, all of which it overwrites, through the split's path:
// begin(), its layers as add() encodes them, commit() and its completion
// check (layer() in AneFfnMeasurement.cpp). A program it compiles waits for
// the ANE's service until `interrupted` returns true. Each duration passes
// usable() where it is measured.
class Measurement final {
public:
  Measurement(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
              const std::array<metal::MetalBuffer, 2> &hidden, std::function<bool()> interrupted);

  // The split's GPU part alone and its split layer of full chunks with the
  // ANE taking the channel units nearest two shares, and the GPU's FFN
  // alone, on the same four layers spread over the model, whose formats can
  // differ with depth: a program of one function of a full chunk's rows
  // each, which the ANE's service keeps once compiled.
  [[nodiscard]] Timings time();
  // The GPU alone over the least and the most rows of `split`'s functions,
  // and `split` over a chunk of the rows of each of its functions, on the
  // first layers.
  [[nodiscard]] ChunkTimings chunks(AneFfn &split);

private:
  // The per-layer milliseconds of `count` `rows`-row FFN layers of `layers`
  // from the first, fewer than all of them (AneFfnMeasurement.cpp): the
  // GPU's alone without `split`, or `split`'s less `standIn`; and of
  // `split`'s GPU part alone of full chunks.
  [[nodiscard]] double layer(std::span<const SwiGluProjections> layers, uint32_t count, uint32_t rows,
                             AneFfn *split = nullptr, double standIn = 0.0);
  [[nodiscard]] double gpuPart(std::span<const SwiGluProjections> layers, AneFfn &split);

  metal::MetalBackend &backend_;
  std::span<const SwiGluProjections> layers_;
  PrefillFfnBuffers ffn_;
  std::array<metal::MetalBuffer, 2> hidden_;
  std::function<bool()> interrupted_;
  const Linear linear_;
};

} // namespace splash::ops::ane_ffn
