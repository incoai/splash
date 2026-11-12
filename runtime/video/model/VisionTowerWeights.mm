#include "video/model/VisionTowerWeights.hpp"

#include "model/SafetensorsCheckpoint.hpp"

#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace splash::video::model {
namespace {

using ops::VisionTower;
const std::string kPrefix = "vision_tower.";

const splash::model::SourceTensor &tensor(const splash::model::SafetensorsCheckpoint &checkpoint,
                                          const std::string &name, const std::vector<uint64_t> &shape) {
  const splash::model::SourceTensor &source = checkpoint.require(kPrefix + name);
  if (source.dtype != "BF16" || source.shape != shape)
    throw std::invalid_argument("vision tower tensor " + name + " is not the unquantized BF16 tower's");
  return source;
}

std::vector<uint16_t> readBf16(const splash::model::SourceTensor &source) {
  std::vector<uint16_t> values(source.bytes / 2);
  source.read(0, {reinterpret_cast<uint8_t *>(values.data()), source.bytes});
  return values;
}

_Float16 toHalf(uint16_t bf16) { return static_cast<_Float16>(std::bit_cast<float>(uint32_t{bf16} << 16)); }

// An fp16 buffer of rows x columns from a [sourceRows][sourceColumns] bf16 tensor, zero beyond it.
metal::MetalBuffer matrix(metal::MetalBackend &backend, const splash::model::SourceTensor &source, uint64_t rows,
                          uint64_t columns, uint64_t sourceRows, uint64_t sourceColumns, const char *label) {
  const std::vector<uint16_t> values = readBf16(source);
  metal::MetalBuffer buffer = backend.allocateBuffer(rows * columns * 2, metal::BufferStorage::Shared, label);
  auto *out = static_cast<_Float16 *>(buffer.contents());
  std::memset(out, 0, rows * columns * 2);
  for (uint64_t r = 0; r < sourceRows; ++r)
    for (uint64_t c = 0; c < sourceColumns; ++c) out[r * columns + c] = toHalf(values[r * sourceColumns + c]);
  return buffer;
}

metal::MetalBuffer vectorBuffer(metal::MetalBackend &backend, const splash::model::SafetensorsCheckpoint &checkpoint,
                          const std::string &name, uint64_t size, uint64_t padded, const char *label) {
  return matrix(backend, tensor(checkpoint, name, {size}), 1, padded, 1, size, label);
}

ops::VisionTowerAffine affine(metal::MetalBackend &backend, const splash::model::SafetensorsCheckpoint &checkpoint,
                              const std::string &name, uint64_t outputs, uint64_t inputs, uint64_t paddedOutputs,
                              uint64_t paddedInputs) {
  return {matrix(backend, tensor(checkpoint, name + ".weight", {outputs, inputs}), paddedOutputs, paddedInputs, outputs,
                 inputs, "vision-tower-weight"),
          vectorBuffer(backend, checkpoint, name + ".bias", outputs, paddedOutputs, "vision-tower-bias")};
}

ops::VisionTowerNorm layerNorm(metal::MetalBackend &backend, const splash::model::SafetensorsCheckpoint &checkpoint,
                               const std::string &name, uint64_t width) {
  return {vectorBuffer(backend, checkpoint, name + ".weight", width, width, "vision-tower-norm"),
          vectorBuffer(backend, checkpoint, name + ".bias", width, width, "vision-tower-norm")};
}

} // namespace

VisionTowerWeights::VisionTowerWeights(metal::MetalBackend &backend,
                                       const splash::model::SafetensorsCheckpoint &checkpoint)
    : backend_(backend), checkpoint_(checkpoint) {
  constexpr uint64_t H = VisionTower::kHidden, I = VisionTower::kIntermediate, M = VisionTower::kMergedHidden,
                     O = VisionTower::kOutput;
  tensor(checkpoint, "patch_embed.proj.weight", {H, 2, 16, 16, 3});
  tensor(checkpoint, "patch_embed.proj.bias", {H});
  tensor(checkpoint, "pos_embed.weight", {VisionTower::kPositionSide * VisionTower::kPositionSide, H});
  for (uint32_t b = 0; b < VisionTower::kDepth; ++b) {
    const std::string p = "blocks." + std::to_string(b) + ".";
    for (const char *n : {"norm1", "norm2"}) {
      tensor(checkpoint, p + n + ".weight", {H});
      tensor(checkpoint, p + n + ".bias", {H});
    }
    for (const auto &[n, out, in] : {std::tuple{"attn.qkv", 3 * H, H}, {"attn.proj", H, H}, {"mlp.linear_fc1", I, H},
                                     {"mlp.linear_fc2", H, I}}) {
      tensor(checkpoint, p + n + ".weight", {out, in});
      tensor(checkpoint, p + n + ".bias", {out});
    }
  }
  for (uint32_t m = 0; m < 4; ++m) {
    const std::string p = m ? "deepstack_merger_list." + std::to_string(m - 1) + "." : std::string("merger.");
    tensor(checkpoint, p + "norm.weight", {m ? M : H});
    tensor(checkpoint, p + "norm.bias", {m ? M : H});
    tensor(checkpoint, p + "linear_fc1.weight", {M, M});
    tensor(checkpoint, p + "linear_fc1.bias", {M});
    tensor(checkpoint, p + "linear_fc2.weight", {O, M});
    tensor(checkpoint, p + "linear_fc2.bias", {O});
  }
  if (checkpoint.find(kPrefix + "blocks." + std::to_string(VisionTower::kDepth) + ".norm1.weight") ||
      checkpoint.find(kPrefix + "deepstack_merger_list.3.norm.weight"))
    throw std::invalid_argument("vision tower has more blocks or deepstack mergers than Qwen3-VL-32B's");
}

ops::VisionTowerEmbedding VisionTowerWeights::embedding() const {
  constexpr uint64_t H = VisionTower::kHidden, P = VisionTower::kPatchDimension;
  ops::VisionTowerEmbedding out;
  // MLX's Conv3d weight (out, t, h, w, c) -> (out, c, t, h, w), the patch rows' feature order.
  const std::vector<uint16_t> values = readBf16(tensor(checkpoint_, "patch_embed.proj.weight", {H, 2, 16, 16, 3}));
  out.patch.weight = backend_.allocateBuffer(H * P * 2, metal::BufferStorage::Shared, "vision-tower-patch");
  auto *w = static_cast<_Float16 *>(out.patch.weight.contents());
  for (uint64_t o = 0; o < H; ++o)
    for (uint64_t t = 0; t < 2; ++t)
      for (uint64_t y = 0; y < 16; ++y)
        for (uint64_t x = 0; x < 16; ++x)
          for (uint64_t c = 0; c < 3; ++c)
            w[o * P + ((c * 2 + t) * 16 + y) * 16 + x] = toHalf(values[(((o * 2 + t) * 16 + y) * 16 + x) * 3 + c]);
  out.patch.bias = vectorBuffer(backend_, checkpoint_, "patch_embed.proj.bias", H, H, "vision-tower-bias");
  const uint64_t cells = uint64_t{VisionTower::kPositionSide} * VisionTower::kPositionSide;
  out.positions = matrix(backend_, tensor(checkpoint_, "pos_embed.weight", {cells, H}), cells, H, cells, H,
                         "vision-tower-positions");
  return out;
}

ops::VisionTowerBlock VisionTowerWeights::block(uint32_t index) const {
  constexpr uint64_t H = VisionTower::kHidden, I = VisionTower::kIntermediate, IP = VisionTower::kPaddedIntermediate;
  if (index >= VisionTower::kDepth) throw std::invalid_argument("vision tower block out of range");
  const std::string p = "blocks." + std::to_string(index) + ".";
  return {layerNorm(backend_, checkpoint_, p + "norm1", H),
          affine(backend_, checkpoint_, p + "attn.qkv", 3 * H, H, 3 * H, H),
          affine(backend_, checkpoint_, p + "attn.proj", H, H, H, H),
          layerNorm(backend_, checkpoint_, p + "norm2", H),
          affine(backend_, checkpoint_, p + "mlp.linear_fc1", I, H, IP, H),
          affine(backend_, checkpoint_, p + "mlp.linear_fc2", H, I, H, IP)};
}

ops::VisionTowerMerger VisionTowerWeights::merger(uint32_t index) const {
  constexpr uint64_t H = VisionTower::kHidden, M = VisionTower::kMergedHidden, O = VisionTower::kOutput;
  if (index > VisionTower::kDeepstack.size()) throw std::invalid_argument("vision tower merger out of range");
  const std::string p = index ? "deepstack_merger_list." + std::to_string(index - 1) + "." : std::string("merger.");
  return {layerNorm(backend_, checkpoint_, p + "norm", index ? M : H),
          affine(backend_, checkpoint_, p + "linear_fc1", M, M, M, M),
          affine(backend_, checkpoint_, p + "linear_fc2", O, M, O, M)};
}

} // namespace splash::video::model
