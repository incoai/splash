#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Linear.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::ops {

// The dense FFN of a prefill chunk on the GPU in int8, by the ANE FFN split's
// W8A8 algorithm (ops/AneFfn.hpp): the chunk's normalized rows rotated in
// blocks of ANE_FFN_INPUT_BLOCK and quantized to int8 per token; gate and up
// as int8 matmuls into int32, h = silu(gate) * up; h rotated in blocks of
// ANE_FFN_INTERMEDIATE_BLOCK and quantized to int8 per token; down likewise,
// added to the residual. Each layer's weights are rotated alike and quantized
// to int8 per row from their planes as a command reaches the layer, into one
// set of int8 rows every layer reuses, so the model keeps its weights as they
// are; each row's scale is computed once here. Each GEMM dispatch converts the
// weights the next one multiplies beside its own tiles: gate and up's down,
// down the next layer's gate and up. MPP runs the matmuls on the GPU's int8
// path, which an M5's runs at about twice its bf16 rate
// (kernels/prefill/int8_ffn.metal); an earlier GPU runs it no faster than
// bf16, so the FFN is for an M5 or later only (unsupported()).
class Int8Ffn final {
public:
  // A chunk of fewer rows runs the bf16 kernels: converting the weights costs
  // a pass over each layer's planes whatever the chunk's rows. Against them
  // on Qwen3.8-27B's FFN on an M5 Max, 128 rows take 1.46x as long, 256 0.96x,
  // 512 0.76x and 2048 0.61x.
  static constexpr uint32_t kMinimumRows = 256, kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;

  // Computes the row scales of `layers`, which it takes (unsupported()),
  // whose planes the FFN reads for the life of the object. It runs on any
  // GPU splash does.
  Int8Ffn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers);
  Int8Ffn(const Int8Ffn &) = delete;
  Int8Ffn &operator=(const Int8Ffn &) = delete;

  // Why the FFN does not take `layers`, or null when it does: unrotated
  // quantized projections of one shape in whole rotation blocks.
  [[nodiscard]] static const char *unsupported(std::span<const SwiGluProjections> layers);
  // Why a prefill does not run `layers` in int8 on `device`, or null when it
  // does: on an M5 or later GPU (Apple10), whose int8 matmul runs at about
  // twice its bf16 rate, the layers the FFN takes.
  [[nodiscard]] static const char *unsupported(const DeviceCapabilities &device,
                                               std::span<const SwiGluProjections> layers);
  // The Metal memory of the FFN of `layers`.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers);
  // What Metal allocated for it, which plannedBytes() bounds.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }

  // Whether it takes a chunk of `rows` rows.
  [[nodiscard]] static bool takes(uint32_t rows) noexcept { return rows >= kMinimumRows && rows <= kMaximumRows; }
  // Layer `layer`'s FFN of a chunk of `rows` rows it takes: output = residual
  // + FFN of ffn.normalized, through ffn.intermediate. A command adds its
  // layers in order from layer 0, which converts its own gate and up first,
  // as a layer added out of order does; every layer's down converts the next
  // one's.
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows);

private:
  struct Shape final {
    uint32_t hidden = 0, intermediate = 0, layers = 0;
  };
  // Everything the FFN allocates: the rotation signs, every layer's row
  // scales, the int8 rows of one layer's gate, up and down, and a chunk's
  // int8 inputs and intermediate rows with their per-token scales.
  struct Memory final {
    metal::MetalBuffer signs, rowScales, gate, up, down, inputs, inputScale, intermediate, intermediateScale;
  };
  // A projection's weight planes as the kernels bind them: an image tensor's
  // plane0, plane1 and meta, its groups of 32 inputs per row, and its format.
  struct Planes final {
    explicit Planes(const Projection &projection);
    std::array<metal::MetalBuffer, 3> buffers;
    uint32_t groups = 0, format = 0;
  };
  enum class Matrix : uint8_t { Gate, Up, Down };

  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers);
  template <class MakeBuffer> [[nodiscard]] static Memory allocate(const Shape &shape, MakeBuffer &&buffer);
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, Matrix matrix) const;
  // Layer `layer`'s down over the chunk's `rows` intermediate rows,
  // converting layer `next`'s gate and up beside it when there is one: with
  // `rows` 0, only the conversion.
  void addDown(metal::CommandGraph &graph, uint32_t layer, uint32_t next, metal::MetalBuffer residual,
               metal::MetalBuffer output, uint32_t rows) const;

  metal::MetalBackend &backend_;
  Shape shape_;
  Memory memory_;
  std::vector<std::array<Planes, 3>> layers_;
  uint64_t allocatedBytes_ = 0;
  // The layer whose gate and up the last down converted.
  uint32_t nextLayer_ = 0;
};

} // namespace splash::ops
