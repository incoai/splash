// Times the bf16 decode projections of the Omni models' shapes
// (kernels/omni/gemm_bf16.metal omni_gemv_bf16*): the GEMV instances
// DenseLinear runs (one or two columns a simdgroup, compiled for one, two or
// four rows) and, from two rows, the 32x64 MPP tile.
//   omni-dense-bench METALLIB [--rows N]
//   omni-dense-bench METALLIB --real MODEL_DIR [--prefix P] [--head NAME|none]
//                    [--rows N] [--rounds N] [--policy] [--filter TEXT]
//   omni-dense-bench METALLIB --tower [ROWS,ROWS...] [--conv | --decoder] [--rounds N] [--filter A|B...]
// --tower (or --rows above eight) times the audio tower's GEMMs with their
// epilogues at 151, 166, 300 and 500 tokens (--conv adds the convolutions'
// im2col GEMMs; --decoder takes the bf16 decoder's prefill projections
// instead): DenseLinear's policy, every production MPP tile and, in a
// METALLIB linked with dev/benchmarks/omni_dense_variants.metal (make
// omni-dense-variants), their variants, each checked against an fp64
// reference and the policy's bytes, with useful and padded TFLOPS.
// Each configuration is timed as a chain of 16 dispatches in one command
// (best of 8 after 4 warm-up chains), cycling weight copies of at least
// 512 MB so every dispatch streams its weights from DRAM.
//
// --real takes the decoder projections under P (default "model.") and the
// head NAME (default the tied model.embed_tokens.weight) of a bf16 Qwen3
// checkpoint (Qwen3-ASR-1.7B-bf16: QKV fused 4096x2048, O, gate/up, down,
// the 151936x2048 head into fp32; the TTS code predictor under
// talker.code_predictor.model. with --head
// talker.code_predictor.lm_head.0.weight), successive layers as the copies, and
// times the plain weights against the packed ones (omni_gemv_packed_*),
// interleaved over rounds (median and best of the rounds' bests), with the
// bf16 GEMV's other column count at one row. Two or four columns a
// simdgroup of packed weights, timed while choosing, ran slower at every
// row count.
// It checks that every packed weight unpacks to its bits and that each
// packed kernel writes the plain kernel's output bytes.
//
// --real with --rows 5..16, or 2..16 with --policy, times instead the
// DenseLinear policy over the bf16 weights alone and with the packed copy
// (the MMA tiles, with split scratch), and each MMA tile unsplit and split
// in two, every one checked against an fp64 reference and for split
// counters left set.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Omni.h"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"
#include "ops/Omni.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;

namespace {

std::mt19937 rng(11);

metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, uint64_t count, float scale, const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(count * 2, metal::BufferStorage::Shared, label);
  auto *bits = static_cast<uint16_t *>(buffer.contents());
  std::uniform_real_distribution<float> value(-scale, scale);
  for (uint64_t i = 0; i < count; ++i) {
    const float f = value(rng);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    bits[i] = static_cast<uint16_t>(u >> 16);
  }
  return buffer;
}

struct Shape {
  const char *name;
  uint32_t n, k;
  bool gateUp, residual, f32;
};

double timeChain(metal::MetalBackend &backend, const std::function<void(metal::CommandGraph &)> &build) {
  double best = 1e9;
  for (int i = 0; i < 12; ++i) {
    metal::CommandGraph graph;
    for (int j = 0; j < 16; ++j) build(graph);
    const metal::CommandTiming timing = backend.submitCommandAsync(graph.command()).wait();
    if (i >= 4) best = std::min(best, timing.gpuSeconds * 1e3 / 16);
  }
  return best;
}

// A bf16 weight of the checkpoint: the named tensors stacked along N.
std::vector<uint16_t> readRows(const model::SafetensorsCheckpoint &checkpoint, const std::vector<std::string> &names) {
  std::vector<uint16_t> rows;
  for (const std::string &name : names) {
    const model::SourceTensor &tensor = checkpoint.require(name);
    if (tensor.dtype != "BF16") throw std::runtime_error(name + " is not bf16");
    const size_t at = rows.size();
    rows.resize(at + tensor.bytes / 2);
    tensor.read(0, std::span<uint8_t>(reinterpret_cast<uint8_t *>(rows.data() + at), tensor.bytes));
  }
  return rows;
}

// Packs a weight over the load threads, in ranges of 64 rows.
std::vector<uint8_t> pack(const std::vector<uint16_t> &weights, uint32_t n, uint32_t k) {
  const uint32_t ranges = n / 64;
  std::vector<uint32_t> escapes(ranges + 1, 0);
  model::parallelFor(ranges, [&](size_t r, unsigned) {
    escapes[r + 1] = ops::omni::packedEscapeBlocks(std::span(weights).subspan(r * 64 * k, uint64_t{64} * k), k);
  });
  for (uint32_t r = 0; r < ranges; ++r) escapes[r + 1] += escapes[r];
  std::vector<uint8_t> packed(ops::omni::packedDenseBytes(n, k, escapes[ranges]));
  model::parallelFor(ranges, [&](size_t r, unsigned) {
    ops::omni::packDenseRows(weights, n, k, uint32_t(r * 64), 64, escapes[r], packed);
  });
  return packed;
}

metal::MetalBuffer bytesBuffer(metal::MetalBackend &backend, const void *bytes, uint64_t size, const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(size, metal::BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), bytes, size);
  return buffer;
}

uint32_t floatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

float bf16Value(uint16_t bits) {
  const uint32_t u = uint32_t(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A way to project a step of 2-16 rows: DenseLinear's policy over the bf16
// weights alone and with the packed copy, or one MMA tile at a split count.
struct TileVariant {
  enum class Kind { Policy, Kernel } kind;
  std::string label, kernel;
  bool packed = false;
  uint32_t columns = 0, splits = 1;
  std::vector<double> ms{};
};

std::vector<TileVariant> tileVariants(uint32_t rows, uint32_t k, bool head, const std::string &filter) {
  std::vector<TileVariant> v{{TileVariant::Kind::Policy, "policy bf16", "", false, 0, 1},
                             {TileVariant::Kind::Policy, "policy packed", "", true, 0, 1}};
  const bool r16 = rows > 8;
  for (const bool packed : {false, true}) {
    if (packed && (r16 || (k / 64) % 2)) continue;
    for (const uint32_t splits : {1u, 2u}) {
      if (splits > 1 && (head || (k / 64) % 4)) continue;
      const std::string name = std::string(packed ? "omni_mma_packed" : "omni_mma_bf16") + (r16 ? "_r16" : "_r8");
      v.push_back({TileVariant::Kind::Kernel, name + " k/" + std::to_string(splits), name, packed, r16 ? 64u : 32u,
                   splits});
    }
  }
  if (!filter.empty())
    std::erase_if(v, [&](const TileVariant &t) { return t.label.find(filter) == std::string::npos; });
  return v;
}

// Rows 2-16 of the real weights: the policy and the MMA tiles,
// interleaved over rounds, each checked against an fp64 reference on the
// first copy.
int realTiles(metal::MetalBackend &backend, const model::SafetensorsCheckpoint &checkpoint,
              const std::vector<std::string> &shapeNames, const std::vector<bool> &gateUps,
              const std::vector<bool> &residuals, const std::vector<bool> &f32s,
              const std::function<std::vector<std::string>(size_t shape, uint32_t copy, bool gate)> &tensors,
              uint32_t layers, uint32_t rows, uint32_t rounds, const std::string &filter) {
  std::printf("rows %u; per-dispatch us (median / best of %u rounds), GB/s of the bf16 bytes\n", rows, rounds);
  for (size_t si = 0; si < shapeNames.size(); ++si) {
    const bool gateUp = gateUps[si], f32 = f32s[si], head = f32;
    uint32_t n = 0, k = 0;
    for (const std::string &name : tensors(si, 0, true)) {
      const model::SourceTensor &tensor = checkpoint.require(name);
      n += uint32_t(tensor.shape[0]);
      k = uint32_t(tensor.shape[1]);
    }
    const uint64_t elements = uint64_t{n} * k;
    const uint32_t streams = gateUp ? 2 : 1;
    // Stream 0 holds the gate, stream 1 the up projection.
    std::vector<metal::MetalBuffer> plain[2], packed[2];
    std::vector<uint16_t> first[2];
    uint64_t packedBytes = 0;
    // Successive layers until the packed copies pass 512 MB (the head: two).
    for (uint32_t copy = 0; copy < 2 || packedBytes < 512e6; ++copy)
      for (uint32_t s = 0; s < streams; ++s) {
        std::vector<uint16_t> weights = readRows(checkpoint, tensors(si, copy % layers, s == 0));
        if (weights.size() != elements) throw std::runtime_error(shapeNames[si] + " has another shape");
        const std::vector<uint8_t> bytes = pack(weights, n, k);
        plain[s].push_back(bytesBuffer(backend, weights.data(), elements * 2, "plain"));
        packed[s].push_back(bytesBuffer(backend, bytes.data(), bytes.size(), "packed"));
        packedBytes += bytes.size();
        if (copy == 0) first[s] = std::move(weights);
      }
    const uint32_t copies = uint32_t(plain[0].size());
    const double plainBytes = double(elements) * 2 * streams;
    std::printf("%s %ux%u%s: %u copies, packed %.4f of the bytes\n", shapeNames[si].c_str(), n, k,
                gateUp ? " x2" : "", copies, double(packedBytes) / copies / plainBytes);
    metal::MetalBuffer input = bf16Buffer(backend, uint64_t{32} * k, 1.0F, "input");
    metal::MetalBuffer residual = bf16Buffer(backend, uint64_t{rows} * n, 1.0F, "residual");
    metal::MetalBuffer output = backend.allocateBuffer(uint64_t{rows} * n * 4, metal::BufferStorage::Shared, "out");
    const ops::omni::DenseSplitScratch scratch = ops::omni::denseSplitScratch(n, k, 16);
    metal::MetalBuffer partials = backend.allocateBuffer(
        std::max<uint64_t>(scratch.partialsBytes, uint64_t{2} * 2 * rows * n * 4), metal::BufferStorage::Shared, "partials");
    metal::MetalBuffer counters = backend.allocateBuffer(uint64_t{n} * 4, metal::BufferStorage::Shared, "counters");
    std::memset(counters.contents(), 0, counters.sizeBytes());
    OmniGemmParams p{};
    p.rows = rows;
    p.output_size = n;
    p.input_size = k;
    p.flags = (gateUp ? OMNI_GEMM_FLAG_GATE_UP : 0u) | (residuals[si] ? OMNI_GEMM_FLAG_RESIDUAL : 0u);
    p.input_stride = k;
    p.output_stride = n;
    const ops::omni::GemmOptions options{ops::omni::Activation::None, residuals[si],
                                         f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16};
    const std::string suffix = f32 ? "_f32" : "";
    std::vector<TileVariant> variants = tileVariants(rows, k, head, filter);
    const auto add = [&](metal::CommandGraph &g, const TileVariant &v, uint32_t c) {
      const uint32_t u = gateUp ? 1 : 0;
      switch (v.kind) {
      case TileVariant::Kind::Policy: {
        ops::omni::DenseWeights up{plain[u][c], {}, n, k}, gate{plain[0][c], {}, n, k};
        if (v.packed) {
          up.packed = packed[u][c];
          gate.packed = packed[0][c];
        }
        const ops::omni::GemmBuffers buffers{input, output, residual, 0, 0, partials, counters};
        if (gateUp) ops::omni::DenseLinear::addGateUp(g, gate, up, buffers, rows, options);
        else ops::omni::DenseLinear::add(g, up, buffers, rows, options);
        break;
      }
      case TileVariant::Kind::Kernel: {
        const auto &w = v.packed ? packed : plain;
        g.add(v.kernel + (gateUp ? "_gu" : "") + suffix,
              {input, w[u][c], w[0][c], output, output, residual, output, partials, counters}, p,
              {n / v.columns, v.splits, 1}, {OMNI_MMA_THREADS, 1, 1});
        break;
      }
      }
    };
    // The fp64 reference of the first copy.
    std::vector<double> reference(uint64_t{rows} * n);
    const auto *x = static_cast<const uint16_t *>(input.contents());
    const auto *res = static_cast<const uint16_t *>(residual.contents());
    model::parallelFor(n, [&](size_t column, unsigned) {
      for (uint32_t row = 0; row < rows; ++row) {
        double sum = 0, gate = 0;
        for (uint32_t i = 0; i < k; ++i) {
          const double xi = bf16Value(x[uint64_t{row} * k + i]);
          sum += xi * bf16Value(first[gateUp ? 1 : 0][column * k + i]);
          if (gateUp) gate += xi * bf16Value(first[0][column * k + i]);
        }
        if (gateUp) sum = gate / (1.0 + std::exp(-gate)) * sum;
        if (residuals[si]) sum += bf16Value(res[uint64_t{row} * n + column]);
        reference[uint64_t{row} * n + column] = sum;
      }
    });
    double scale = 0;
    for (double r : reference) scale = std::max(scale, std::fabs(r));
    std::vector<std::string> checks;
    for (TileVariant &v : variants) {
      metal::CommandGraph check;
      add(check, v, 0);
      std::memset(output.contents(), 0xff, output.sizeBytes());
      (void)backend.submitCommandAsync(check.command()).wait();
      double worst = 0;
      for (uint64_t i = 0; i < reference.size(); ++i) {
        const double got = f32 ? double(static_cast<const float *>(output.contents())[i])
                               : double(bf16Value(static_cast<const uint16_t *>(output.contents())[i]));
        worst = std::max(worst, std::fabs(got - reference[i]));
      }
      const auto *count = static_cast<const uint32_t *>(counters.contents());
      bool clean = true;
      for (uint32_t i = 0; i < n; ++i) clean &= count[i] == 0;
      char line[96];
      std::snprintf(line, sizeof(line), "max error %.2e of the largest output%s", worst / scale,
                    clean ? "" : ", COUNTERS LEFT SET");
      checks.push_back(line);
    }
    for (uint32_t round = 0; round < rounds; ++round)
      for (TileVariant &v : variants) {
        uint32_t turn = 0;
        v.ms.push_back(timeChain(backend, [&](metal::CommandGraph &g) { add(g, v, turn++ % copies); }));
      }
    for (size_t i = 0; i < variants.size(); ++i) {
      TileVariant &v = variants[i];
      std::sort(v.ms.begin(), v.ms.end());
      const double median = v.ms[v.ms.size() / 2];
      std::printf("  %-26s %8.1f / %8.1f us %5.0f GB/s  %s\n", v.label.c_str(), median * 1e3, v.ms.front() * 1e3,
                  plainBytes / median / 1e6, checks[i].c_str());
    }
  }
  return 0;
}

int realWeights(metal::MetalBackend &backend, const std::string &modelDir, const std::string &prefix,
                const std::string &head, uint32_t rows, uint32_t rounds, const std::string &filter, bool policy) {
  const model::SafetensorsCheckpoint checkpoint(modelDir);
  struct RealShape {
    const char *name;
    bool gateUp, residual, f32;
    std::function<std::vector<std::string>(uint32_t layer, bool gate)> tensors;
    uint32_t n = 0, k = 0;
  };
  const auto layer = [&](uint32_t l, const char *suffix) {
    return prefix + "layers." + std::to_string(l) + "." + suffix;
  };
  uint32_t layers = 0;
  while (checkpoint.find(layer(layers, "mlp.down_proj.weight"))) ++layers;
  if (!layers) throw std::runtime_error("no decoder layers under " + prefix);
  std::vector<RealShape> shapes{
      {"qkv", false, false, false,
       [&](uint32_t l, bool) {
         return std::vector<std::string>{layer(l, "self_attn.q_proj.weight"), layer(l, "self_attn.k_proj.weight"),
                                         layer(l, "self_attn.v_proj.weight")};
       }},
      {"o +res", false, true, false,
       [&](uint32_t l, bool) { return std::vector<std::string>{layer(l, "self_attn.o_proj.weight")}; }},
      {"gate/up", true, false, false,
       [&](uint32_t l, bool gate) {
         return std::vector<std::string>{layer(l, gate ? "mlp.gate_proj.weight" : "mlp.up_proj.weight")};
       }},
      {"down +res", false, true, false,
       [&](uint32_t l, bool) { return std::vector<std::string>{layer(l, "mlp.down_proj.weight")}; }},
  };
  if (head != "none")
    shapes.push_back({"head f32", false, false, true, [&](uint32_t, bool) { return std::vector<std::string>{head}; }});
  // Shapes as the checkpoint stores them (QKV stacked).
  for (RealShape &shape : shapes)
    for (const std::string &name : shape.tensors(0, true)) {
      const model::SourceTensor &tensor = checkpoint.require(name);
      if (tensor.shape.size() != 2) throw std::runtime_error(name + " is not a matrix");
      shape.n += uint32_t(tensor.shape[0]);
      shape.k = uint32_t(tensor.shape[1]);
    }
  if (rows >= 5 || policy) {
    std::vector<std::string> names;
    std::vector<bool> gateUps, residuals, f32s;
    for (const RealShape &shape : shapes) {
      names.push_back(shape.name);
      gateUps.push_back(shape.gateUp);
      residuals.push_back(shape.residual);
      f32s.push_back(shape.f32);
    }
    return realTiles(backend, checkpoint, names, gateUps, residuals, f32s,
                     [&](size_t s, uint32_t l, bool gate) { return shapes[s].tensors(l, gate); }, layers, rows, rounds,
                     filter);
  }
  uint32_t r = 1;
  while (r < rows) r *= 2;
  std::printf("rows %u (kernels for up to %u); per-dispatch ms (median / best of %u rounds), GB/s streamed, and "
              "the plain bf16 bytes' rate\n",
              rows, r, rounds);
  for (const RealShape &shape : shapes) {
    const uint64_t elements = uint64_t{shape.n} * shape.k;
    const uint32_t streams = shape.gateUp ? 2 : 1;
    std::vector<metal::MetalBuffer> plain[2], packed[2];
    uint64_t packedBytes = 0, escapeBlocks = 0;
    // Successive layers until the packed copies pass 512 MB (the head: two).
    for (uint32_t copy = 0; copy < 2 || packedBytes < 512e6; ++copy) {
      for (uint32_t s = 0; s < streams; ++s) {
        const std::vector<uint16_t> weights = readRows(checkpoint, shape.tensors(copy % layers, s == 0));
        if (weights.size() != elements) throw std::runtime_error(std::string(shape.name) + " has another shape");
        const std::vector<uint8_t> bytes = pack(weights, shape.n, shape.k);
        if (copy < layers) {
          std::vector<uint16_t> back(elements);
          ops::omni::unpackDense(bytes, shape.n, shape.k, back);
          if (back != weights) throw std::runtime_error(std::string(shape.name) + " does not unpack to its bits");
          escapeBlocks += (bytes.size() - ops::omni::packedDenseBytes(shape.n, shape.k, 0)) / OMNI_PACKED_ESCAPE_BYTES;
        }
        plain[s].push_back(bytesBuffer(backend, weights.data(), elements * 2, "plain"));
        packed[s].push_back(bytesBuffer(backend, bytes.data(), bytes.size(), "packed"));
        packedBytes += bytes.size();
      }
    }
    const uint32_t copies = uint32_t(plain[0].size());
    const double plainBytes = double(elements) * 2 * streams;
    const double packedPerCall = double(packedBytes) / copies;
    std::printf("%s %ux%u%s: %u copies, packed %.4f of the bytes, escapes %.4f%% of blocks\n", shape.name, shape.n,
                shape.k, shape.gateUp ? " x2" : "", copies, packedPerCall / plainBytes,
                100.0 * escapeBlocks / (double(std::min(copies, layers)) * streams * elements / 64));
    metal::MetalBuffer input = bf16Buffer(backend, uint64_t{std::max(rows, 4u)} * shape.k, 1.0F, "input");
    metal::MetalBuffer residual = bf16Buffer(backend, uint64_t{rows} * shape.n, 1.0F, "residual");
    metal::MetalBuffer bias = backend.allocateBuffer(uint64_t{shape.n} * 4, metal::BufferStorage::Shared, "bias");
    std::memset(bias.contents(), 0, bias.sizeBytes());
    metal::MetalBuffer output =
        backend.allocateBuffer(uint64_t{rows} * shape.n * 4, metal::BufferStorage::Shared, "out");
    OmniGemmParams p{};
    p.rows = rows;
    p.output_size = shape.n;
    p.input_size = shape.k;
    p.flags = (shape.gateUp ? OMNI_GEMM_FLAG_GATE_UP : 0u) | (shape.residual ? OMNI_GEMM_FLAG_RESIDUAL : 0u);
    p.input_stride = shape.k;
    p.output_stride = shape.n;
    struct Variant {
      std::string kernel;
      uint32_t columns;
      bool packed;
      std::vector<double> ms;
    };
    // The bf16 policy's columns (ops/Omni.cpp denseTile), the packed
    // kernel (one column), then at one row the bf16 kernel's other columns.
    const uint32_t policy = rows == 1 && !shape.gateUp ? 2 : 1;
    const auto name = [&](const char *kernel, uint32_t c) {
      return std::string(kernel) + "_c" + std::to_string(c) + "_r" + std::to_string(r) + (shape.f32 ? "_f32" : "");
    };
    std::vector<Variant> variants{{name("omni_gemv_bf16", policy), policy, false, {}},
                                  {name("omni_gemv_packed", 1), 1, true, {}}};
    if (r == 1) variants.push_back({name("omni_gemv_bf16", 3 - policy), 3 - policy, false, {}});
    const auto grid = [&](uint32_t columns) {
      return metal::DispatchSize{(shape.n + 8 * columns - 1) / (8 * columns), 1, 1};
    };
    std::vector<uint8_t> reference;
    for (Variant &v : variants) {
      const auto &w = v.packed ? packed : plain;
      metal::CommandGraph check;
      check.add(v.kernel, {input, w[0][0], shape.gateUp ? w[1][0] : w[0][0], bias, bias, residual, output}, p,
                grid(v.columns), {256, 1, 1});
      (void)backend.submitCommandAsync(check.command()).wait();
      const auto *out = static_cast<const uint8_t *>(output.contents());
      const std::vector<uint8_t> bytes(out, out + uint64_t{rows} * shape.n * (shape.f32 ? 4 : 2));
      if (reference.empty()) reference = bytes;
      if (bytes != reference)
        std::printf("  %s: output differs from %s\n", v.kernel.c_str(), variants[0].kernel.c_str());
    }
    for (uint32_t round = 0; round < rounds; ++round)
      for (Variant &v : variants) {
        const auto &w = v.packed ? packed : plain;
        uint32_t turn = 0;
        v.ms.push_back(timeChain(backend, [&](metal::CommandGraph &g) {
          const uint32_t c = turn++ % copies;
          g.add(v.kernel, {input, w[0][c], shape.gateUp ? w[1][c] : w[0][c], bias, bias, residual, output}, p,
                grid(v.columns), {256, 1, 1});
        }));
      }
    for (Variant &v : variants) {
      std::sort(v.ms.begin(), v.ms.end());
      const double median = v.ms[v.ms.size() / 2];
      std::printf("  %-32s %8.4f / %8.4f ms %6.0f GB/s (bf16 %4.0f)\n", v.kernel.c_str(), median, v.ms.front(),
                  (v.packed ? packedPerCall : plainBytes) / median / 1e6, plainBytes / median / 1e6);
    }
  }
  return 0;
}

// The audio tower's GEMMs (ops/OmniAudioEncoder.cpp) with their epilogues:
// conv2 and conv3 over im2col rows of a group of eight chunks and of four
// (12 chunks, 151 tokens: one of each), conv_out, a layer's QKV, out, fc1
// and fc2 (24 layers), proj1 and proj2 at the token count. The convolutions
// pad 480 channels to 512 and 4320 taps to 4352; useful FLOPs count 480
// and 4320. --decoder takes instead the bf16 Qwen3 decoder's prefill
// projections (28 layers; the residual stream bf16).
struct TowerShape {
  const char *name;
  uint32_t n, k, usefulN, usefulK;
  bool bias, residual, f32, gateUp;
  ops::omni::Activation activation;
  uint32_t fixedRows; // 0: the token count
  uint32_t count;     // dispatches a file
};

// A way to run a shape: DenseLinear's policy, one production MPP tile
// (gemm_bf16.metal omni_gemm_bf16_*), or a variant of
// dev/benchmarks/omni_dense_variants.metal.
struct TowerVariant {
  enum class Kind { Policy, Production, Variant } kind;
  std::string name;
  uint32_t tileM, tileN, bk, simdgroups, barrier;
  bool pretransposed, half;
  std::vector<double> ms{};
};

std::vector<TowerVariant> towerVariants(const std::string &filter) {
  std::vector<TowerVariant> v{{TowerVariant::Kind::Policy, "policy", 0, 0, 0, 0, 0, false, false}};
  const uint32_t production[][3] = {{32, 64, 8}, {32, 128, 4}, {64, 64, 4}, {64, 64, 8}, {64, 128, 4}, {64, 128, 8}};
  for (const auto &t : production)
    v.push_back({TowerVariant::Kind::Production,
                 "omni_gemm_bf16_m" + std::to_string(t[0]) + "n" + std::to_string(t[1]) + "s" + std::to_string(t[2]),
                 t[0], t[1], 0, t[2], 0, false, false});
  struct Spec {
    uint32_t bk, barrier;
    bool pretransposed, half;
  };
  const Spec specs[] = {{32, 1, false, false},  {64, 1, false, false},  {128, 1, false, false}, {256, 1, false, false},
                        {512, 1, false, false}, {0, 0, false, false},   {64, 0, false, false},  {64, 4, false, false},
                        {128, 0, false, false}, {256, 0, false, false}, {64, 1, true, false},   {256, 0, true, false},
                        {64, 1, false, true},   {256, 0, false, true},  {0, 0, false, true}};
  const uint32_t tiles[][3] = {{32, 64, 4},  {32, 64, 8},  {64, 64, 4},   {64, 64, 8},   {64, 128, 4},
                               {64, 128, 8}, {32, 128, 4}, {128, 64, 8}, {128, 128, 8}, {32, 32, 4}};
  for (const auto &t : tiles)
    for (const Spec &s : specs) {
      const std::string name = "omni_tv_m" + std::to_string(t[0]) + "n" + std::to_string(t[1]) + "k" +
                               std::to_string(s.bk) + "s" + std::to_string(t[2]) + "b" + std::to_string(s.barrier) +
                               (s.pretransposed ? "p" : "t") + (s.half ? "hf" : "bf");
      v.push_back({TowerVariant::Kind::Variant, name, t[0], t[1], s.bk, t[2], s.barrier, s.pretransposed, s.half});
    }
  // The filter keeps the policy and the ways holding any of its
  // |-separated parts.
  std::vector<std::string> parts;
  for (std::string rest = filter; !rest.empty();) {
    const size_t bar = rest.find('|');
    parts.push_back(rest.substr(0, bar));
    rest = bar == std::string::npos ? "" : rest.substr(bar + 1);
  }
  if (!parts.empty())
    std::erase_if(v, [&](const TowerVariant &t) {
      return t.kind != TowerVariant::Kind::Policy &&
             std::none_of(parts.begin(), parts.end(), [&](const std::string &p) { return t.name.find(p) != std::string::npos; });
    });
  return v;
}

uint16_t halfBits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

int towerSweep(metal::MetalBackend &backend, const std::vector<uint32_t> &tokenCounts, uint32_t rounds,
               const std::string &filter, bool conv, bool decoder) {
  using ops::omni::Activation;
  std::vector<TowerShape> shapes;
  if (decoder) {
    shapes = {{"qkv", 4096, 2048, 4096, 2048, false, false, false, false, Activation::None, 0, 28},
              {"o +res", 2048, 2048, 2048, 2048, false, true, false, false, Activation::None, 0, 28},
              {"gate/up", 6144, 2048, 6144, 2048, false, false, false, true, Activation::None, 0, 28},
              {"down +res", 2048, 6144, 2048, 6144, false, true, false, false, Activation::None, 0, 28}};
  } else {
    if (conv)
      shapes = {{"conv2 8 chunks", 512, 4352, 480, 4320, true, false, false, false, Activation::Gelu, 6400, 1},
                {"conv2 4 chunks", 512, 4352, 480, 4320, true, false, false, false, Activation::Gelu, 3200, 1},
                {"conv3 8 chunks", 512, 4352, 480, 4320, true, false, false, false, Activation::Gelu, 1664, 1},
                {"conv3 4 chunks", 512, 4352, 480, 4320, true, false, false, false, Activation::Gelu, 832, 1}};
    for (const TowerShape &s : std::initializer_list<TowerShape>{
             {"conv_out f32", 1024, 7680, 1024, 7680, false, false, true, false, Activation::None, 0, 1},
             {"qkv +bias", 3072, 1024, 3072, 1024, true, false, false, false, Activation::None, 0, 24},
             {"out +bias +res f32", 1024, 1024, 1024, 1024, true, true, true, false, Activation::None, 0, 24},
             {"fc1 +bias gelu", 4096, 1024, 4096, 1024, true, false, false, false, Activation::Gelu, 0, 24},
             {"fc2 +bias +res f32", 1024, 4096, 1024, 4096, true, true, true, false, Activation::None, 0, 24},
             {"proj1 +bias gelu", 1024, 1024, 1024, 1024, true, false, false, false, Activation::Gelu, 0, 1},
             {"proj2 +bias", 2048, 1024, 2048, 1024, true, false, false, false, Activation::None, 0, 1}})
      shapes.push_back(s);
  }
  const std::vector<TowerVariant> all = towerVariants(filter);
  std::printf("per-dispatch us (median of %u rounds of the best of 8 chains of 16), useful / padded TFLOPS; err: max "
              "|out - fp64| over the largest output; = : the policy's bits\n",
              rounds);
  for (const TowerShape &shape : shapes) {
    const uint64_t elements = uint64_t{shape.n} * shape.k;
    const uint32_t streams = shape.gateUp ? 2 : 1;
    const uint32_t copies = std::max<uint32_t>(2, uint32_t(512e6 / (double(elements) * 2 * streams)) + 1);
    // One random weight (and gate), copied so every dispatch streams its own
    // from DRAM; the variants' copies of the same values pretransposed
    // ([K][N]) and as fp16.
    metal::MetalBuffer first = bf16Buffer(backend, elements, 0.05F, "w");
    metal::MetalBuffer firstGate = shape.gateUp ? bf16Buffer(backend, elements, 0.05F, "g") : first;
    const auto *bits = static_cast<const uint16_t *>(first.contents());
    const auto *gateBits = static_cast<const uint16_t *>(firstGate.contents());
    const bool variants = !shape.gateUp && std::any_of(all.begin(), all.end(), [](const TowerVariant &v) {
      return v.kind == TowerVariant::Kind::Variant;
    });
    std::vector<uint16_t> transposed, halves, halfTransposed;
    if (variants) {
      transposed.resize(elements);
      halves.resize(elements);
      halfTransposed.resize(elements);
      for (uint32_t n = 0; n < shape.n; ++n)
        for (uint32_t k = 0; k < shape.k; ++k) transposed[uint64_t{k} * shape.n + n] = bits[uint64_t{n} * shape.k + k];
      for (uint64_t i = 0; i < elements; ++i) {
        halves[i] = halfBits(bf16Value(bits[i]));
        halfTransposed[i] = halfBits(bf16Value(transposed[i]));
      }
    }
    // Weight copies by layout: [N][K] bf16, [K][N] bf16, [N][K] fp16, [K][N]
    // fp16, and the gate's.
    std::vector<metal::MetalBuffer> weights[5];
    for (uint32_t c = 0; c < copies; ++c) {
      weights[0].push_back(c ? bytesBuffer(backend, bits, elements * 2, "w") : first);
      if (shape.gateUp) weights[4].push_back(c ? bytesBuffer(backend, gateBits, elements * 2, "g") : firstGate);
      if (!variants) continue;
      weights[1].push_back(bytesBuffer(backend, transposed.data(), elements * 2, "wt"));
      weights[2].push_back(bytesBuffer(backend, halves.data(), elements * 2, "wh"));
      weights[3].push_back(bytesBuffer(backend, halfTransposed.data(), elements * 2, "wht"));
    }
    std::vector<float> biasValues(shape.n);
    std::uniform_real_distribution<float> biasValue(-0.5F, 0.5F);
    for (float &b : biasValues) b = biasValue(rng);
    metal::MetalBuffer bias = bytesBuffer(backend, biasValues.data(), uint64_t{shape.n} * 4, "bias");
    std::vector<uint32_t> rowCounts;
    if (shape.fixedRows) rowCounts = {shape.fixedRows};
    else rowCounts = tokenCounts;
    for (const uint32_t rows : rowCounts) {
      // Whole tiles of 128 rows: every tile reads its padding.
      const uint64_t inputRows = (rows + 127) / 128 * 128;
      metal::MetalBuffer input = bf16Buffer(backend, inputRows * shape.k, 1.0F, "input");
      std::vector<uint16_t> inputHalves(variants ? inputRows * shape.k : 0);
      for (uint64_t i = 0; i < inputHalves.size(); ++i)
        inputHalves[i] = halfBits(bf16Value(static_cast<const uint16_t *>(input.contents())[i]));
      metal::MetalBuffer inputHalf =
          variants ? bytesBuffer(backend, inputHalves.data(), inputHalves.size() * 2, "input half") : input;
      const uint64_t element = shape.f32 ? 4 : 2;
      const uint64_t outputBytes = uint64_t{rows} * shape.n * element;
      // Residual rows as the engine keeps them (fp32 or bf16), added in place.
      std::vector<float> residualValues(uint64_t{rows} * shape.n);
      std::uniform_real_distribution<float> residualValue(-2.0F, 2.0F);
      for (float &r : residualValues) r = shape.f32 ? residualValue(rng) : bf16Value(uint16_t(floatBits(residualValue(rng)) >> 16));
      std::vector<uint16_t> residualBf16(residualValues.size());
      for (size_t i = 0; i < residualValues.size(); ++i) residualBf16[i] = uint16_t(floatBits(residualValues[i]) >> 16);
      const void *residualSource = shape.f32 ? static_cast<const void *>(residualValues.data()) : residualBf16.data();
      metal::MetalBuffer hidden = bytesBuffer(backend, residualSource, outputBytes, "hidden");
      metal::MetalBuffer residual = bytesBuffer(backend, residualSource, outputBytes, "residual");
      metal::MetalBuffer output = backend.allocateBuffer(outputBytes, metal::BufferStorage::Shared, "out");
      OmniGemmParams p{};
      p.rows = rows;
      p.output_size = shape.n;
      p.input_size = shape.k;
      p.flags = (shape.bias ? OMNI_GEMM_FLAG_BIAS : 0u) | (shape.residual ? OMNI_GEMM_FLAG_RESIDUAL : 0u) |
                (shape.gateUp ? OMNI_GEMM_FLAG_GATE_UP : 0u);
      p.activation = static_cast<uint32_t>(shape.activation);
      p.input_stride = shape.k;
      p.output_stride = shape.n;
      const std::string suffix = std::string(shape.gateUp ? "_gu" : "") + (shape.f32 ? "_f32" : "");
      const ops::omni::GemmOptions options{shape.activation, shape.residual,
                                           shape.f32 ? ops::FloatOutput::Float32 : ops::FloatOutput::BFloat16};
      std::vector<TowerVariant> ways;
      for (const TowerVariant &v : all)
        if (v.kind == TowerVariant::Kind::Policy ||
            (shape.n % v.tileN == 0 && !(shape.gateUp && v.kind == TowerVariant::Kind::Variant)))
          ways.push_back(v);
      const auto add = [&](metal::CommandGraph &g, const TowerVariant &v, uint32_t c, bool inPlace) {
        const metal::MetalBuffer &res = inPlace ? hidden : residual;
        const metal::MetalBuffer &out = inPlace && shape.residual ? hidden : output;
        const metal::MetalBuffer &gate = shape.gateUp ? weights[4][c] : weights[0][c];
        const metal::DispatchSize grid{(rows + v.tileM - 1) / v.tileM, shape.n / std::max(v.tileN, 1u), 1};
        switch (v.kind) {
        case TowerVariant::Kind::Policy: {
          const ops::omni::DenseWeights up{weights[0][c], shape.bias ? bias : metal::MetalBuffer{}, shape.n, shape.k};
          const ops::omni::DenseWeights gateWeights{gate, shape.bias ? bias : metal::MetalBuffer{}, shape.n, shape.k};
          const ops::omni::GemmBuffers buffers{input, out, res, 0, 0};
          if (shape.gateUp) ops::omni::DenseLinear::addGateUp(g, gateWeights, up, buffers, rows, options);
          else ops::omni::DenseLinear::add(g, up, buffers, rows, options);
          break;
        }
        case TowerVariant::Kind::Production:
          g.add(v.name + suffix, {input, weights[0][c], gate, bias, bias, res, out}, p, grid, {v.simdgroups * 32, 1, 1});
          break;
        case TowerVariant::Kind::Variant: {
          const metal::MetalBuffer &w = weights[(v.half ? 2 : 0) + (v.pretransposed ? 1 : 0)][c];
          g.add(v.name + suffix, {v.half ? inputHalf : input, w, bias, res, out}, p, grid, {v.simdgroups * 32, 1, 1});
          break;
        }
        }
      };
      // fp64 references of the first and last 64 rows, from the bf16 values.
      std::vector<uint32_t> checkRows;
      for (uint32_t r = 0; r < std::min(rows, 64u); ++r) checkRows.push_back(r);
      for (uint32_t r = std::max(64u, rows >= 64 ? rows - 64 : 0); r < rows; ++r) checkRows.push_back(r);
      std::vector<double> reference(checkRows.size() * shape.n);
      const auto *x = static_cast<const uint16_t *>(input.contents());
      model::parallelFor(shape.n, [&](size_t column, unsigned) {
        for (size_t i = 0; i < checkRows.size(); ++i) {
          const uint32_t row = checkRows[i];
          double sum = 0, gated = 0;
          for (uint32_t k = 0; k < shape.k; ++k) {
            const double xk = bf16Value(x[uint64_t{row} * shape.k + k]);
            sum += xk * bf16Value(bits[column * shape.k + k]);
            if (shape.gateUp) gated += xk * bf16Value(gateBits[column * shape.k + k]);
          }
          if (shape.bias) sum += biasValues[column];
          if (shape.gateUp) sum = gated / (1.0 + std::exp(-gated)) * sum;
          if (shape.activation == Activation::Gelu) sum = 0.5 * sum * (1.0 + std::erf(sum / std::sqrt(2.0)));
          if (shape.residual) sum += residualValues[uint64_t{row} * shape.n + column];
          reference[i * shape.n + column] = sum;
        }
      });
      double scale = 0;
      for (double r : reference) scale = std::max(scale, std::fabs(r));
      std::vector<uint8_t> policyBytes;
      std::string policyKernel;
      std::vector<std::string> checks(ways.size());
      std::vector<bool> usable(ways.size(), true);
      for (size_t i = 0; i < ways.size(); ++i) {
        // The policy first (index 0), so the others compare with its bytes.
        metal::CommandGraph check;
        add(check, ways[i], 0, false);
        if (i == 0) policyKernel = check.dispatches().front().pipelineName;
        std::memset(output.contents(), 0xff, output.sizeBytes());
        try {
          (void)backend.submitCommandAsync(check.command()).wait();
        } catch (const std::exception &error) {
          usable[i] = false;
          checks[i] = std::string("FAILED: ") + error.what();
          continue;
        }
        const auto *out = static_cast<const uint8_t *>(output.contents());
        const std::vector<uint8_t> bytes(out, out + outputBytes);
        if (i == 0) policyBytes = bytes;
        double worst = 0;
        for (size_t c = 0; c < checkRows.size(); ++c)
          for (uint32_t n = 0; n < shape.n; ++n) {
            const uint64_t at = uint64_t{checkRows[c]} * shape.n + n;
            const double got = shape.f32 ? double(reinterpret_cast<const float *>(out)[at])
                                         : double(bf16Value(reinterpret_cast<const uint16_t *>(out)[at]));
            worst = std::max(worst, std::fabs(got - reference[c * shape.n + n]));
          }
        char line[96];
        std::snprintf(line, sizeof(line), "err %.2e %s", worst / scale, bytes == policyBytes ? "=" : " ");
        checks[i] = line;
      }
      for (uint32_t round = 0; round < rounds; ++round)
        for (size_t i = 0; i < ways.size(); ++i) {
          if (!usable[i]) continue;
          uint32_t turn = 0;
          ways[i].ms.push_back(
              timeChain(backend, [&](metal::CommandGraph &g) { add(g, ways[i], turn++ % copies, true); }));
        }
      const double useful = 2.0 * rows * shape.usefulN * shape.usefulK * streams;
      std::printf("%s %ux%u%s, %u rows (x%u a file), %u copies; policy %s\n", shape.name, shape.n, shape.k,
                  shape.gateUp ? " x2" : "", rows, shape.count, copies, policyKernel.c_str());
      const auto median = [&](size_t i) {
        std::vector<double> ms = ways[i].ms;
        std::sort(ms.begin(), ms.end());
        return ms.empty() ? 1e9 : ms[ms.size() / 2];
      };
      std::vector<size_t> order(ways.size());
      for (size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return median(a) < median(b); });
      const double policyMs = median(0);
      for (size_t i : order) {
        const TowerVariant &v = ways[i];
        if (!usable[i]) {
          std::printf("  %-30s %s\n", v.name.c_str(), checks[i].c_str());
          continue;
        }
        const double ms = median(i);
        const uint32_t tileM = v.tileM ? v.tileM : 1;
        const double padded = 2.0 * ((rows + tileM - 1) / tileM * tileM) * shape.n * shape.k * streams;
        std::printf("  %-30s %8.1f us %5.1f / %5.1f TFLOPS %+7.1f us a file  %s\n", v.name.c_str(), ms * 1e3,
                    useful / ms / 1e9, v.tileM ? padded / ms / 1e9 : 0.0, (ms - policyMs) * 1e3 * shape.count,
                    checks[i].c_str());
      }
    }
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: omni-dense-bench METALLIB [--rows N]\n");
    return 2;
  }
  uint32_t rows = 1, rounds = 5;
  std::string realModel, prefix = "model.", head = "model.embed_tokens.weight", filter;
  bool policy = false, conv = false, decoder = false;
  std::vector<uint32_t> towerRows;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--policy") policy = true;
    if (std::string(argv[i]) == "--conv") conv = true;
    if (std::string(argv[i]) == "--decoder") decoder = true;
    if (std::string(argv[i]) == "--tower") {
      towerRows = {151, 166, 300, 500};
      if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0]))) {
        towerRows.clear();
        for (std::string list = argv[i + 1]; !list.empty();) {
          const size_t comma = list.find(',');
          towerRows.push_back(std::stoul(list.substr(0, comma)));
          list = comma == std::string::npos ? "" : list.substr(comma + 1);
        }
      }
    }
  }
  for (int i = 2; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--rows") rows = std::stoul(argv[i + 1]);
    if (std::string(argv[i]) == "--rounds") rounds = std::stoul(argv[i + 1]);
    if (std::string(argv[i]) == "--real") realModel = argv[i + 1];
    if (std::string(argv[i]) == "--prefix") prefix = argv[i + 1];
    if (std::string(argv[i]) == "--head") head = argv[i + 1];
    if (std::string(argv[i]) == "--filter") filter = argv[i + 1];
  }
  if (!rows || rows > 2048) {
    std::fprintf(stderr, "omni dense bench: --rows takes 1..2048\n");
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    if (!realModel.empty()) {
      if (rows > 16 || !rounds) throw std::invalid_argument("--real takes --rows 1..16 and --rounds 1 or more");
      return realWeights(backend, realModel, prefix, head, rows, rounds, filter, policy);
    }
    if (!towerRows.empty() || rows > 8) {
      if (towerRows.empty()) towerRows = {rows};
      return towerSweep(backend, towerRows, std::max(rounds, 1u), filter, conv, decoder);
    }
    const std::vector<Shape> decode{
        {"talker/asr qkv", 4096, 2048, false, false, false},
        {"talker/asr o +res", 2048, 2048, false, true, false},
        {"talker/asr gate/up", 6144, 2048, true, false, false},
        {"talker/asr down +res", 2048, 6144, false, true, false},
        {"talker head f32", 3072, 2048, false, false, true},
        {"predictor qkv", 4096, 1024, false, false, false},
        {"predictor o +res", 1024, 2048, false, true, false},
        {"predictor gate/up", 3072, 1024, true, false, false},
        {"predictor down +res", 1024, 3072, false, true, false},
        {"predictor head f32", 2048, 1024, false, false, true},
    };
    const std::vector<Shape> &shapes = decode;
    uint32_t r = 1;
    while (r < rows) r *= 2;
    std::printf("rows %u (kernels for up to %u); per-dispatch ms and GB/s of weights\n", rows, r);
    for (const Shape &shape : shapes) {
      const uint64_t elements = uint64_t{shape.n} * shape.k;
      const double bytes = double(elements) * 2 * (shape.gateUp ? 2 : 1);
      const uint32_t copies = std::max<uint32_t>(2, uint32_t(512e6 / bytes) + 1);
      std::vector<metal::MetalBuffer> ws, gs;
      for (uint32_t c = 0; c < copies; ++c) {
        ws.push_back(bf16Buffer(backend, elements, 0.05F, "w"));
        if (shape.gateUp) gs.push_back(bf16Buffer(backend, elements, 0.05F, "g"));
      }
      // Whole 64-row tiles: the MPP tiles read them.
      const uint64_t inputRows = (std::max<uint32_t>(rows, 32) + 63) / 64 * 64;
      metal::MetalBuffer input = bf16Buffer(backend, inputRows * shape.k, 1.0F, "input");
      metal::MetalBuffer residual = bf16Buffer(backend, uint64_t{rows} * shape.n, 1.0F, "residual");
      metal::MetalBuffer bias = backend.allocateBuffer(uint64_t{shape.n} * 4, metal::BufferStorage::Shared, "bias");
      std::memset(bias.contents(), 0, bias.sizeBytes());
      metal::MetalBuffer output = backend.allocateBuffer(uint64_t{rows} * shape.n * 4, metal::BufferStorage::Shared, "out");
      OmniGemmParams p{};
      p.rows = rows;
      p.output_size = shape.n;
      p.input_size = shape.k;
      p.flags = (shape.gateUp ? OMNI_GEMM_FLAG_GATE_UP : 0u) | (shape.residual ? OMNI_GEMM_FLAG_RESIDUAL : 0u);
      p.input_stride = shape.k;
      p.output_stride = shape.n;
      std::vector<std::vector<uint8_t>> results;
      const auto run = [&](const std::string &kernel, uint32_t columns, uint32_t tileColumns = 0) {
        const uint32_t tileRows = tileColumns == 128 ? 64 : 32;
        const metal::DispatchSize grid = tileColumns ? metal::DispatchSize{(rows + tileRows - 1) / tileRows, shape.n / tileColumns, 1}
                                                 : metal::DispatchSize{(shape.n + 8 * columns - 1) / (8 * columns), 1, 1};
        uint32_t turn = 0;
        const double ms = timeChain(backend, [&](metal::CommandGraph &g) {
          const metal::MetalBuffer &w = ws[turn % copies];
          const metal::MetalBuffer &gate = shape.gateUp ? gs[turn % copies] : w;
          g.add(kernel + (shape.f32 ? "_f32" : ""), {input, w, gate, bias, bias, residual, output}, p, grid, {256, 1, 1});
          ++turn;
        });
        // Checks every variant against the first on one fixed copy.
        metal::CommandGraph check;
        check.add(kernel + (shape.f32 ? "_f32" : ""), {input, ws[0], shape.gateUp ? gs[0] : ws[0], bias, bias, residual, output}, p,
                  grid, {256, 1, 1});
        (void)backend.submitCommandAsync(check.command()).wait();
        const auto *out = static_cast<const uint8_t *>(output.contents());
        results.emplace_back(out, out + uint64_t{rows} * shape.n * (shape.f32 ? 4 : 2));
        bool same = results.back() == results.front();
        double worst = 0.0;
        if (!same && !shape.f32) {
          const auto bf = [](const uint8_t *b, uint64_t i) { uint32_t u = uint32_t(b[2 * i] | (b[2 * i + 1] << 8)) << 16; float f; std::memcpy(&f, &u, 4); return f; };
          for (uint64_t i = 0; i < uint64_t{rows} * shape.n; ++i)
            worst = std::max(worst, double(std::fabs(bf(results.back().data(), i) - bf(results.front().data(), i))));
        }
        const double flops = 2.0 * rows * shape.n * shape.k * (shape.gateUp ? 2 : 1);
        std::printf("%-22s %-24s %7.4f ms %6.0f GB/s %5.1f TFLOPS", shape.name, kernel.c_str(), ms, bytes / ms / 1e6,
                    flops / ms / 1e9);
        if (!same) std::printf("  differs (max %.3g)", worst);
        std::printf("\n");
      };
      if (r <= 4) {
        run("omni_gemv_bf16_c1_r" + std::to_string(r), 1);
        if (r == 1) run("omni_gemv_bf16_c2_r1", 2);
      }
      if (rows > 1) run("omni_gemm_bf16_m32n64s8", 0, 64);
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "omni dense bench: %s\n", error.what());
    return 1;
  }
  return 0;
}
