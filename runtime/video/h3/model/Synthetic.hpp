#pragma once

// Deterministic synthetic weights and inputs (dev/tests/engine/video_synthetic_metal_test.mm): no weight files,
// the same shapes and memory as the converted VDN-H3 DiT, values from the hash of metal/abi/SyntheticABI.h. Tensor
// seeds: syntheticSeed(group, id), group = block index for block tensors, kIoGroup for the io weights, kInputGroup
// for the request inputs. Value ranges keep the 50-block forward finite (unit-gain linears, gates ~0.1).
#include "video/h3/engine/Encoder.hpp"
#include "metal/abi/video/SyntheticABI.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace h3::model {

inline constexpr int kSyntheticIoGroup = 1000, kSyntheticInputGroup = 2000;

uint64_t syntheticSeed(int group, int id);
float syntheticFloat(uint64_t seed, uint64_t i, float scale, float offset);
// Host arrays: fp32 / bf16 bits
void syntheticF32(uint64_t seed, size_t n, float scale, float offset, float *out);
void syntheticBf16(uint64_t seed, size_t n, float scale, float offset, uint16_t *out);

// GPU fills batched into one command.
class SyntheticFiller {
public:
  explicit SyntheticFiller(metal::MetalBackend &backend) : b_(backend) {}
  [[nodiscard]] metal::MetalBuffer allocate(size_t count, int type, uint64_t seed, float scale, float offset,
                                            std::string_view what);
  void fill(const metal::MetalBuffer &buffer, size_t offsetBytes, size_t count, int type, uint64_t seed, float scale,
            float offset);
  void run();
  [[nodiscard]] size_t bytes() const { return bytes_; }

private:
  struct Job {
    metal::MetalBuffer buffer;
    size_t offset;
    SyntheticFill fill;
  };
  metal::MetalBackend &b_;
  std::vector<Job> jobs_;
  size_t bytes_ = 0;
};

} // namespace h3::model
