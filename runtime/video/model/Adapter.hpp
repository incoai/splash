#pragma once

// Runtime LoRA adapters of the DiT (format h3-lora-v1, converted once at installation from a PEFT adapter
// by the installer): for every adapted linear y = W x + scale B (A x), the two extra matmuls run beside the
// int8 GEMM (ops::Lora, kernels/video/dit/lora.metal) and the add is fused into the GEMM's epilogue, before its QK-norm,
// SwiGLU or gated residual. The adapter's A and B stay resident in bf16 (Pika H3 Flex's FL2VA: rank 128, 1.33 GB).
// Without an adapter nothing of this runs and the DiT's output is bit-identical to the engine's without the feature.
//
// An adapter directory holds manifest.json (format, source, rank, alpha, scale, files) and, per transformer block,
// blocks.N.lora.{qkv,out,fc1,fc2}.{a,b} (A rotated by the int8 linears' Hadamard transform, so that it reads their
// int8 inputs; B [out, r] in the int8 weights' row order: qkv [q | k | v] head-major, fc1 [gate | value]); the QKFIX
// blocks also qkv.a_plain, unrotated, for their fp16 q / k / v GEMM; and the token refiner's
// refiner.N.{qkv,out,fc1,fc2}.{a,b}, run by the prep in bf16 as PEFT does (video/model/Prep.mm).
//
// Which adapters a DiT runs: its manifest.json's "adapters": [{"path": P, "scale": S}] (P relative to the DiT's
// directory or absolute; S multiplies the adapter's own alpha / r; entries that are not objects, such as VDN-H3's
// names of the adapters merged at conversion, are no runtime adapters), or the engine's --adapter PATH[:SCALE] in
// their place.
#include "video/metal/Encoder.hpp"
#include "video/model/WeightFiles.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace splash::video::model {

struct AdapterSpec {
  std::filesystem::path path;
  double scale = 1.0;
};
// The adapters of the DiT in ditDir: `override` when set, else its manifest's "adapters" (none when absent).
[[nodiscard]] std::vector<AdapterSpec> adapterSpecs(const std::filesystem::path &ditDir,
                                                    const std::optional<std::vector<AdapterSpec>> &override);
// PATH or PATH:SCALE of the engine's --adapter.
[[nodiscard]] AdapterSpec parseAdapterSpec(const std::string &text);

// One adapted linear: A bf16 [segments x rank][K] (rotated for the int8 GEMMs), B bf16 [N][rank]; segment s of N
// (qkv: q, k, v of 7168 rows each) reads columns [s rank, (s + 1) rank) of A x.
struct LoraLinear {
  metal::MetalBuffer a, b;
  int32_t segments = 1, n = 0, k = 0;
};

struct AdapterBlock {
  LoraLinear qkv, out, fc1, fc2;
  metal::MetalBuffer qkvPlain;  // QKFIX blocks: A of q / k / v, unrotated (the fp16 GEMM's input)
};

class Adapter {
public:
  // Reads the adapter's DiT tensors (blocks) into resident buffers, qkv.a_plain of the DiT's QKFIX blocks
  // (DitWeights::qkfixBlocks); the refiner's stay on disk for the prep.
  static std::unique_ptr<Adapter> load(metal::MetalBackend &backend, const AdapterSpec &spec,
                                       const std::set<int> &qkfixBlocks);

  [[nodiscard]] const AdapterBlock &block(int b) const { return blocks_.at(b); }
  [[nodiscard]] int32_t rank() const { return rank_; }
  [[nodiscard]] float scale() const { return scale_; }  // alpha / rank x the spec's scale
  [[nodiscard]] size_t residentBytes() const { return residentBytes_; }
  [[nodiscard]] const std::string &name() const { return name_; }

  // The token refiner's tensors (refiner.N.*), from the adapter's directory: bf16 bits, or empty when it has none.
  struct RefinerLinear {
    std::vector<uint16_t> a, b;  // A [rA][K], B [N][rA] (qkv: rA = 3 rank, B block-diagonal on qkv_proj's rows)
    int32_t rows = 0;            // rA
  };
  struct RefinerBlock {
    RefinerLinear qkv, out, fc1, fc2;
  };
  [[nodiscard]] static std::vector<RefinerBlock> refiner(const AdapterSpec &spec, int blocks, float *scale);

private:
  std::vector<AdapterBlock> blocks_;
  int32_t rank_ = 0;
  float scale_ = 1.0f;
  size_t residentBytes_ = 0;
  std::string name_;
};

} // namespace splash::video::model
