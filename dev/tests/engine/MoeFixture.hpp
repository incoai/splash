#pragma once

// A GGUF sparse MoE block for the MoE tests (gguf_moe_test.mm, the kernels):
// weights in GGUF formats with GGML's values of them
// (GgufFormatReference.hpp), bf16 rows, and the buffers a plan runs in.

#include "GgufFormatReference.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ExpertStore.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe_fixture {

// The most rows a test's MoE step runs: a prefill chunk whose tiles end
// ragged.
inline constexpr uint32_t kMaximumRows = 263;

// The tests' weights and rows, the same in every run.
inline std::mt19937 rng(0x35);

inline float bf16(double value) { return float(__bf16(float(value))); }

inline splash::metal::MetalBuffer upload(splash::metal::MetalBackend &backend, const void *data, uint64_t bytes,
                                         const char *label) {
  splash::metal::MetalBuffer buffer = backend.allocateBuffer(bytes, splash::metal::BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

inline splash::metal::MetalBuffer zeros(splash::metal::MetalBackend &backend, uint64_t bytes, const char *label) {
  if (!bytes) return {};
  splash::metal::MetalBuffer buffer = backend.allocateBuffer(bytes, splash::metal::BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// The segments of a repacked [rows, K] tensor in format f and of [rows, K] floats.
inline splash::ops::QuantizedSegment planeSegment(splash::metal::MetalBackend &backend, gguf_reference::Fmt f,
                                                  const gguf_reference::Packed &planes, uint32_t rows, uint32_t K) {
  return splash::ops::QuantizedSegment::planes(
      f, rows, K, upload(backend, planes.w0.data(), planes.w0.size(), "gguf-plane0"),
      kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1.data(), planes.w1.size(), "gguf-plane1")
                                    : splash::metal::MetalBuffer{},
      upload(backend, planes.meta.data(), planes.meta.size(), "gguf-meta"));
}
inline splash::ops::QuantizedSegment floatSegment(splash::metal::MetalBackend &backend,
                                                  const std::vector<float> &values, uint32_t rows, uint32_t K) {
  return splash::ops::QuantizedSegment::floats(rows, K, upload(backend, values.data(), values.size() * sizeof(float),
                                                               "gguf-floats"));
}

// A GGUF tensor [rows, K]: its segment and GGML's fp32 values of it.
struct Tensor {
  splash::ops::QuantizedSegment segment;
  std::vector<float> values;
  uint32_t rows = 0, columns = 0;
  [[nodiscard]] const float *row(uint64_t n) const { return values.data() + n * columns; }
};

inline Tensor quantized(splash::metal::MetalBackend &backend, gguf_reference::Fmt f, uint32_t rows, uint32_t K) {
  const std::vector<uint8_t> native = gguf_reference::makeNative(f, rows, K, rng);
  Tensor t;
  t.rows = rows;
  t.columns = K;
  t.segment = planeSegment(backend, f, gguf_reference::repack(f, native, rows, K, &t.values), rows, K);
  return t;
}

inline Tensor floating(splash::metal::MetalBackend &backend, uint32_t rows, uint32_t K, float scale) {
  std::normal_distribution<float> normal(0.0f, scale);
  Tensor t;
  t.rows = rows;
  t.columns = K;
  t.values.resize(uint64_t{rows} * K);
  for (float &v : t.values) v = normal(rng);
  t.segment = floatSegment(backend, t.values, rows, K);
  return t;
}

// bf16 rows in [-1, 1].
inline std::vector<float> activations(uint64_t elements) {
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::vector<float> values(elements);
  for (float &v : values) v = bf16(unit(rng));
  return values;
}

inline splash::metal::MetalBuffer bfloatBuffer(splash::metal::MetalBackend &backend, const std::vector<float> &values,
                                               const char *label) {
  std::vector<__bf16> bits(values.begin(), values.end());
  return upload(backend, bits.data(), bits.size() * 2, label);
}

// A block of `shape`: its router, shared expert scalar gate, routed experts'
// and shared expert's gate, up and down, their weights with expert e in slot
// e, and the formats of the six projections.
struct Model {
  splash::ops::MoeShape shape;
  Tensor router, sharedGate;
  std::array<Tensor, 3> routed;   // gate, up, down: experts * N rows
  std::array<Tensor, 3> shared;
  splash::ops::BlockMoeWeights weights;
  std::array<gguf_reference::Fmt, 6> formats{};
};

// The formats whose weights `tile` takes, in id order: the register tile takes
// those of quant_register_format.
inline std::vector<gguf_reference::Fmt> tileFormats(splash::ops::MoeGgufTile tile) {
  std::vector<gguf_reference::Fmt> formats;
  for (int f = 0; f < gguf_reference::FMT_COUNT; ++f)
    if (tile == splash::ops::MoeGgufTile::Staged || quant_register_format(uint32_t(f)))
      formats.push_back(gguf_reference::Fmt(f));
  return formats;
}

// The first of `formats` from index i on, cyclically, that `tile` takes in
// rows of K values: one whose column unit divides K (a GGUF stores no
// 256-value blocks in rows of 640, Qwen3.8-Flash-Next's expert down
// projections, which take 32- or 64-value blocks, and preparation takes
// PQ2_0's 128-value blocks and the MLX affine formats' groups of 128 only in
// rows of whole superblocks), and on the register tile one it takes in such
// rows (quant_register_tensor: 64-value blocks in rows of whole 256 values).
inline gguf_reference::Fmt fitting(const std::vector<gguf_reference::Fmt> &formats, size_t i, uint32_t K,
                                   splash::ops::MoeGgufTile tile) {
  for (size_t j = 0; j < formats.size(); ++j) {
    const gguf_reference::Fmt g = formats[(i + j) % formats.size()];
    if (K % quant_column_unit(kQuantFormats[g].block_elements) == 0 &&
        (tile == splash::ops::MoeGgufTile::Staged || quant_register_tensor(g, K)))
      return g;
  }
  throw std::logic_error("no format fits the row");
}

// The model of `shape` and format f that `tile` runs: gate, up and down in f
// and the next two formats the tile takes, cyclically, and the shared
// expert's in the three after them, each the next that fits its rows; with
// upInGateFormats, up in the gate's formats.
inline Model makeModel(splash::metal::MetalBackend &backend, const splash::ops::MoeShape &shape, int f,
                       bool upInGateFormats, splash::ops::MoeGgufTile tile = splash::ops::MoeGgufTile::Staged) {
  const std::vector<gguf_reference::Fmt> formats = tileFormats(tile);
  const size_t first = std::find(formats.begin(), formats.end(), gguf_reference::Fmt(f)) - formats.begin();
  const uint32_t hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  const uint32_t n[3] = {intermediate, intermediate, hidden}, k[3] = {hidden, hidden, intermediate};
  Model m;
  m.shape = shape;
  for (size_t i = 0; i < m.formats.size(); ++i) m.formats[i] = fitting(formats, first + i, k[i % 3], tile);
  if (upInGateFormats) {
    m.formats[1] = m.formats[0];
    m.formats[4] = m.formats[3];
  }
  m.router = floating(backend, shape.experts, hidden, 0.05f);
  m.sharedGate = floating(backend, 1, hidden, 0.05f);
  for (int p = 0; p < 3; ++p) {
    m.routed[p] = quantized(backend, m.formats[p], shape.experts * n[p], k[p]);
    m.shared[p] = quantized(backend, m.formats[3 + p], n[p], k[p]);
  }
  splash::ops::BlockMoeWeights gguf;
  gguf.router = m.router.segment;
  gguf.sharedScalarGate = m.sharedGate.segment;
  gguf.gate = {m.routed[0].segment, m.shared[0].segment};
  gguf.up = {m.routed[1].segment, m.shared[1].segment};
  gguf.down = {m.routed[2].segment, m.shared[2].segment};
  gguf.slotTable = splash::model::residentExpertSlots(backend, shape.experts);
  gguf.slots = shape.experts;
  m.weights = gguf;
  return m;
}

// The model's six formats, "gate/up/down/shared gate/shared up/shared down".
inline std::string formatNames(const Model &m) {
  std::string names;
  for (const gguf_reference::Fmt format : m.formats)
    names += std::string(names.empty() ? "" : "/") + gguf_reference::fmtName(format);
  return names;
}

// A step's buffers, and the fp32 values of its input and residual rows.
struct Buffers {
  splash::ops::MoeBuffers moe;
  std::vector<float> input, residual;
};

// The scratch `plan` runs in, zeroed.
inline void allocate(splash::metal::MetalBackend &backend, Buffers &b, const splash::ops::MoePlan &plan) {
  const auto &w = plan.workspace();
  for (const splash::ops::MoeScratchField &field : splash::ops::kMoeScratchFields)
    b.moe.scratch.*field.buffer = zeros(backend, w.*field.bytes, "moe-scratch");
}

} // namespace moe_fixture
